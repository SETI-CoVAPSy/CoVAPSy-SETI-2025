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

        # Follow-the-gap tuning
        self.FOLLOW_GAP_WINDOW_DEG = 60    # forward search window (degrees, total)
        self.SAFETY_DISTANCE       = 0.35  # minimum acceptable clearance (m)
        self.BUBBLE_DEG            = 10    # obstacle 'bubble' half-width in degrees

        # CHANGED: Adapted KP for the "Real Car" logic.
        # Real car used 0.02 deg/mm. Converted to rad/m, that's roughly 0.35.
        # You can tune this around 0.3 to 0.6.
        self.KP               = 0.3

        self.MAX_STEER        = 1.0   # maximum steering angle (rad)
        self.INVERT_STEERING  = False # Set to True if steering is inverted
        # ==========================================================

        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.publisher_ = self.create_publisher(AckermannDrive, '/car/command', 10)


    def lidar_callback(self, msg: LaserScan):
        """Follow-the-gap controller.
        """
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # sanitize measurements
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        ranges = np.clip(ranges, 0.0, msg.range_max)

        # focus on forward window (centered on 0)
        half_window_rad = np.radians(self.FOLLOW_GAP_WINDOW_DEG / 2.0)
        window_mask = np.abs(angles) <= half_window_rad
        window_ranges = ranges[window_mask].copy()
        window_angles = angles[window_mask]

        drive_msg = AckermannDrive()

        if window_ranges.size == 0:
            drive_msg.steering_angle = 0.0
            drive_msg.speed = self.SLOW_SPEED
            self.publisher_.publish(drive_msg)
            return

        # mark as free when clearance > SAFETY_DISTANCE
        free_mask = window_ranges > self.SAFETY_DISTANCE

        # apply a small angular bubble around close obstacles to avoid edge grazing
        if not np.all(free_mask):
            bubble_rad = np.radians(self.BUBBLE_DEG)
            close_idxs = np.where(~free_mask)[0]
            for ci in close_idxs:
                low_ang = window_angles[ci] - bubble_rad
                high_ang = window_angles[ci] + bubble_rad
                removal = (window_angles >= low_ang) & (window_angles <= high_ang)
                free_mask[removal] = False

        # find contiguous free segments inside the forward window
        if not np.any(free_mask):
            # no gap found — be conservative
            best_angle = 0.0
            drive_msg.speed = self.SLOW_SPEED
        else:
            idxs = np.where(free_mask)[0]
            groups = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
            # choose the largest gap
            best_group = max(groups, key=lambda g: g.size)
            # inside that gap, pick the index with maximum clearance (safer) — can pick middle instead
            rel_idx = best_group[np.argmax(window_ranges[best_group])]
            best_angle = float(window_angles[rel_idx])
            # speed proportional to average clearance in chosen gap (clamped)
            avg_clearance = float(np.mean(window_ranges[best_group]))
            drive_msg.speed = float(np.clip((avg_clearance / self.FRONT_THRESHOLD) * self.TARGET_SPEED,
                                            self.SLOW_SPEED, self.TARGET_SPEED))

        # steering: proportional to angle (radians)
        steering = best_angle * self.KP
        if self.INVERT_STEERING:
            steering = -steering
        drive_msg.steering_angle = float(np.clip(steering, -self.MAX_STEER, self.MAX_STEER))

        self.publisher_.publish(drive_msg)
    

def main(args=None):
    rclpy.init(args=args)
    node = CircuitFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()