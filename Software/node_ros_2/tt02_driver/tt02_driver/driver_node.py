#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32

from tt02_driver.gilbert_driver_webots import GilbertDriverWebots


class TT02DriverNode(Node):
    """ROS 2 driver node controlling the TT-02 in Webots."""

    def __init__(self):
        super().__init__("tt02_driver")

        self.driver = GilbertDriverWebots(verbose=True)
        self.webots_driver = self.driver.get_driver() 

        self.target_speed = 0.0     # m/s
        self.target_angle = 0.0     # degrees

        # subscriber to Ackermann commands
        self.sub_cmd = self.create_subscription(
            AckermannDrive,
            "/car/command",
            self.callback_command,
            10
        )

        self.timer = self.create_timer(0.02, self.update)

        self.get_logger().info("TT02 driver started.")

    def callback_command(self, msg: AckermannDrive):
        self.target_speed = msg.speed
        self.target_angle = msg.steering_angle * 180.0 / 3.14159265358979

        self.target_speed = max(self.driver.SPEED_LIMIT_REVERSE, min(self.driver.SPEED_LIMIT_FORWARD, self.target_speed))
        self.target_angle = max(-self.driver.ANGLE_LIMIT_DEG, min(self.driver.ANGLE_LIMIT_DEG, self.target_angle))
        
    
    def update(self):
        step_result = self.webots_driver.step()

        if step_result == -1:
            # Webots simulation ended (window closed)
            self.get_logger().warn("Simulation ended.")
            rclpy.shutdown()
            return
        
        self.driver.set_speed_mps(self.target_speed)
        self.driver.set_steering_angle_deg(self.target_angle)



def main(args=None):
    rclpy.init(args=args) 
    node = TT02DriverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()