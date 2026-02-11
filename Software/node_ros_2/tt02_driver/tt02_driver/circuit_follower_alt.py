"""
Aternative implementation (more complex PID control)
"""

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
        self.TARGET_SPEED     = 0.7  # Speed (m/s)
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


    def lidar_callback(self, msg: LaserScan):
        # self.get_logger().info(f"Mid: {lidar_angle_middle}, Angle: {lidar_angle_amplitude}")
        lidar_angle_amplitude = msg.angle_max - msg.angle_min
        lidar_angle_middle = (msg.angle_max + msg.angle_min) / 2.0
        point_count = len(msg.ranges)

        drive_msg = AckermannDrive()

        # Get ranges at some angles
        def get_range_at_angle(angle_deg):
            angle_rad = np.radians(angle_deg)
            index = int(((angle_rad - msg.angle_min) / (msg.angle_max - msg.angle_min)) * point_count)
            index = np.clip(index, 0, point_count - 1)
            return msg.ranges[index]
        range_m30 = get_range_at_angle(-30)
        range_m15 = get_range_at_angle(-15)
        range_0   = get_range_at_angle(0)
        range_p15 = get_range_at_angle(15)
        range_p30 = get_range_at_angle(30)
        # self.get_logger().info(f"Ranges: m30:{range_m30:.2f} 0:{range_0:.2f} p30:{range_p30:.2f}")
        
        # Mean, ignoring invalid values, default to max range if all invalid
        mean_p = np.mean([r for r in [range_p30, range_p15] if np.isfinite(r)])
        mean_m = np.mean([r for r in [range_m30, range_m15] if np.isfinite(r)])
        if not np.isfinite(mean_p):
            mean_p = msg.range_max
        if not np.isfinite(mean_m):
            mean_m = msg.range_max
        
        # Steering
        if range_0 < mean_p or range_0 < mean_m: # Agressive steering
            if mean_p - mean_m != 0:
                drive_msg.steering_angle = 1/(mean_p - mean_m) * self.KP
            else:
                drive_msg.steering_angle = 0.0
        else: # Proportional steering
            drive_msg.steering_angle = (mean_p - mean_m) * self.KP
        
        drive_msg.steering_angle = np.clip(drive_msg.steering_angle, -self.MAX_STEER, self.MAX_STEER)

        # Speed Control
        front_distance = np.mean([r for r in [range_m15, range_0, range_p15] if np.isfinite(r)])
        drive_msg.speed = np.clip(front_distance**1.5 * self.KP, self.SLOW_SPEED*2, self.TARGET_SPEED)
        self.publisher_.publish(drive_msg)
    

def main(args=None):
    rclpy.init(args=args)
    node = CircuitFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()