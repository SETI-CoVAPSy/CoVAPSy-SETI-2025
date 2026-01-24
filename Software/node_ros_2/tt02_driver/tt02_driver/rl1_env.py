"""
First RL prototype (Environment)

Responsibility:
1. Subscribe to sensors (/scan, /odom) and Actuator feedback (/car/command)
2. Compute Reward (Progress - Smoothness - Crash)
3. Compute State (Normalized Lidar + Velocity)
4. Publish /rl/state, /rl/reward, /rl/done
"""
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32MultiArray, Float32, Bool

class RLEnvironment(Node):
    def __init__(self):
        super().__init__('rl_environment')

        # --- Configuration ---
        self.LIDAR_RAYS = 20        # Downsample 360 rays to 20
        self.MAX_LIDAR_DIST = 10.0  # Max range for normalization
        self.CRASH_DIST = 0.25      # Meters. Below this = Crash
        self.CRASH_PENALTY = -20.0  # Sparse penalty
        
        # --- State Variables ---
        self.current_scan = np.full(self.LIDAR_RAYS, self.MAX_LIDAR_DIST)
        self.current_speed = 0.0
        self.last_steering = 0.0
        
        # --- ROS Interfaces ---
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_cmd  = self.create_subscription(AckermannDrive, '/car/command', self.cmd_callback, 10)

        self.pub_state  = self.create_publisher(Float32MultiArray, '/rl/state', 10)
        self.pub_reward = self.create_publisher(Float32, '/rl/reward', 10)
        self.pub_done   = self.create_publisher(Bool, '/rl/done', 10)

        # Run the environment loop at 20Hz
        self.timer = self.create_timer(0.05, self.step)

    def scan_callback(self, msg: LaserScan):
        # Downsample Lidar: Take N evenly spaced rays
        ranges = np.array(msg.ranges)
        # Handle inf/nan
        ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.MAX_LIDAR_DIST)
        
        # Resample to fixed size
        indices = np.linspace(0, len(ranges)-1, self.LIDAR_RAYS, dtype=int)
        self.current_scan = ranges[indices]

    def odom_callback(self, msg: Odometry):
        self.current_speed = msg.twist.twist.linear.x

    def cmd_callback(self, msg: AckermannDrive):
        # Keep track of last action for smoothness penalty
        self.last_steering = msg.steering_angle

    def step(self):
        # 1. Check Termination (Crash)
        min_dist = np.min(self.current_scan)
        done = False
        if min_dist < self.CRASH_DIST:
            done = True

        # 2. Compute Reward
        # Reward = (Progress) - (Jerky Steering) + (Crash Penalty)
        reward = (self.current_speed * 1.0) - (abs(self.last_steering) * 0.5)
        
        if done:
            reward += self.CRASH_PENALTY

        # 3. Construct State Vector
        # Normalize inputs to roughly [-1, 1] or [0, 1]
        norm_scan = self.current_scan / self.MAX_LIDAR_DIST
        norm_speed = self.current_speed / 8.0 # Assuming max speed ~8m/s
        norm_steer = self.last_steering / 0.35 # Assuming max steer ~0.35 rad

        state = np.concatenate([norm_scan, [norm_speed], [norm_steer]])
        
        # 4. Publish
        state_msg = Float32MultiArray()
        state_msg.data = state.tolist()
        self.pub_state.publish(state_msg)

        reward_msg = Float32()
        reward_msg.data = float(reward)
        self.pub_reward.publish(reward_msg)

        done_msg = Bool()
        done_msg.data = done
        self.pub_done.publish(done_msg)

        if done:
            self.get_logger().info(f"Crash detected! Min Dist: {min_dist:.2f}", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = RLEnvironment()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
