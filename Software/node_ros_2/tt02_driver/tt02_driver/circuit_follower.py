import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
import numpy as np

class CircuitFollower(Node):
    def __init__(self):
        super().__init__('circuit_follower')

        # ==========================================================
        # AUTONOMOUS PILOT PARAMETERS
        # ==========================================================
        self.TARGET_SPEED     = 0.5  # Speed (m/s)
        self.SLOW_SPEED       = 0.3   # speed when near obstacle

        self.FRONT_THRESHOLD  = 1.0   # Distance to slow down

        # CHANGED: Adapted KP for the "Real Car" logic.
        # Real car used 0.02 deg/mm. Converted to rad/m, that's roughly 0.35.
        # You can tune this around 0.3 to 0.6.
        self.KP               = 0.4

        self.MAX_STEER        = 0.7   # maximum steering angle (rad)
        self.INVERT_STEERING  = False # Set to True if steering is inverted
        # ==========================================================

        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.publisher_ = self.create_publisher(AckermannDrive, '/car/command', 10)

    def lidar_callback(self, msg):
        # 1. Data cleanup
        ranges = np.array(msg.ranges)
        ranges = np.where((ranges <= msg.range_min) | (np.isnan(ranges)) | (np.isinf(ranges)),
                          msg.range_max, ranges)

        # 2. Get indices for specific angles (+60° and -60°)
        angle_target_rad = np.radians(15) # 1.047 rad

        # Calculate indices: index = (target_angle - min_angle) / increment
        idx_left  = int((angle_target_rad - msg.angle_min) / msg.angle_increment)
        idx_right = int((-angle_target_rad - msg.angle_min) / msg.angle_increment)

        # Safety check to ensure indices are inside the array
        idx_left  = np.clip(idx_left, 0, len(ranges) - 1)
        idx_right = np.clip(idx_right, 0, len(ranges) - 1)

        # 3. Extract specific distances (Like the real car)
        dist_left_60  = ranges[idx_left]
        dist_right_60 = ranges[idx_right]

        # Get front distance for speed control (taking a small center average)
        mid_idx = len(ranges) // 2
        dist_front = np.mean(ranges[mid_idx-5 : mid_idx+5])

        drive_msg = AckermannDrive()

        # 1. Calculate Steering
        # Formula: Angle = Gain * (Left - Right)
        # If Left > Right (Gap on Left), result is positive -> Turn Left (to center)
        diff = dist_left_60 - dist_right_60
        steering = diff * self.KP

        # 2. Speed Control
        # Simple logic: If obstacle in front, slow down. Otherwise, target speed.
        if dist_front < self.FRONT_THRESHOLD:
             drive_msg.speed = self.SLOW_SPEED
             # Optional: Increase steering effect slightly when slow/avoiding
             steering *= 1.5
        else:
             drive_msg.speed = self.TARGET_SPEED

        # ==========================================================

        # Apply steering
        drive_msg.steering_angle = float(steering)

        # Inversion check
        if self.INVERT_STEERING:
            drive_msg.steering_angle *= -1.0

        # Clip limits
        drive_msg.steering_angle = np.clip(drive_msg.steering_angle, -self.MAX_STEER, self.MAX_STEER)

        self.publisher_.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CircuitFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
