"""
First RL prototype (Agent)

Responsibility:
1. Subscribe to /rl/state, /rl/reward, /rl/done
2. Run TD3 Algorithm (Stable Baselines 3)
3. Publish /car/command
"""
import rclpy
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from std_msgs.msg import Float32MultiArray, Float32, Bool
from ackermann_msgs.msg import AckermannDrive

class TT02GymEnv(gym.Env):
    """
    Custom Gymnasium Environment that bridges ROS 2 topics to RL interface.
    """
    def __init__(self, ros_node):
        super().__init__()
        self.node = ros_node
        
        # --- Hyperparameters ---
        # State: 20 Lidar + 1 Speed + 1 Steering = 22
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        
        # Action: Speed, Steering (Normalized [-1, 1])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Physical limits for denormalization
        self.MAX_SPEED_MPS = 2.0 
        self.MAX_STEER_RAD = 0.35 

        # --- ROS Interfaces ---
        self.sub_state = self.node.create_subscription(Float32MultiArray, '/rl/state', self.state_cb, 10)
        self.sub_reward = self.node.create_subscription(Float32, '/rl/reward', self.reward_cb, 10)
        self.sub_done = self.node.create_subscription(Bool, '/rl/done', self.done_cb, 10)
        self.pub_cmd = self.node.create_publisher(AckermannDrive, '/car/command', 10)
        
        # --- State Buffers ---
        self.latest_state = np.zeros(22, dtype=np.float32)
        self.latest_reward = 0.0
        self.latest_done = False
        
        # Synchronization event
        self.msg_received = threading.Event()

    def state_cb(self, msg):
        self.latest_state = np.array(msg.data, dtype=np.float32)
        self.msg_received.set()

    def reward_cb(self, msg):
        self.latest_reward = msg.data

    def done_cb(self, msg):
        self.latest_done = msg.data

    def step(self, action):
        # 1. Execute Action
        phys_speed = float(action[0]) * self.MAX_SPEED_MPS
        phys_steer = float(action[1]) * self.MAX_STEER_RAD
        
        cmd = AckermannDrive()
        cmd.speed = phys_speed
        cmd.steering_angle = phys_steer
        self.pub_cmd.publish(cmd)
        
        # 2. Wait for next observation (Sync)
        self.msg_received.clear()
        if not self.msg_received.wait(timeout=0.2):
            self.node.get_logger().warn("Timeout waiting for state update!", throttle_duration_sec=5.0)
        
        # 3. Return
        return self.latest_state, self.latest_reward, self.latest_done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Stop the car
        cmd = AckermannDrive()
        cmd.speed = 0.0
        cmd.steering_angle = 0.0
        self.pub_cmd.publish(cmd)
        
        # Wait for fresh state
        self.msg_received.clear()
        self.msg_received.wait(timeout=0.5)
        
        return self.latest_state, {}

def main(args=None):
    rclpy.init(args=args)
    
    # Create ROS node
    ros_node = rclpy.create_node('td3_agent_sb3')
    
    # Spin in background thread so callbacks work while SB3 learns
    spinner_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spinner_thread.start()
    
    # Init Gym Env
    env = TT02GymEnv(ros_node)
    
    # Init TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, learning_rate=1e-3)
    
    ros_node.get_logger().info("Starting TD3 Training...")
    
    try:
        model.learn(total_timesteps=50000, log_interval=10)
        model.save("td3_tt02_final")
    except KeyboardInterrupt:
        pass
    
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
