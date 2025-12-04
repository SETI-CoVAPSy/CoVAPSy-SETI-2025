#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32
from controller import Lidar
from sensor_msgs.msg import LaserScan



from tt02_driver.gilbert_driver_webots import GilbertDriverWebots


class TT02DriverNode(Node):
    """ROS 2 driver node controlling the TT-02 in Webots."""

    def __init__(self):
        super().__init__("tt02_driver")

        self.driver = GilbertDriverWebots(verbose=True)
        self.webots_driver = self.driver.get_driver() 

        self.basicTimeStep = int(self.webots_driver.getBasicTimeStep())
        self.sensorTimeStep = 4 * self.basicTimeStep

        self.lidar = self.webots_driver.getDevice("RpLidarA2")
        self.lidar.enable(self.sensorTimeStep)

        self.nb_rays = self.lidar.getHorizontalResolution()
        self.fov = self.lidar.getFov()

        self.publisher = self.create_publisher(LaserScan, "/scan", 10)

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
        self.publish_scan()

    def publish_scan(self):
        raw = self.lidar.getRangeImage()

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "RpLidarA2"

        # Champs LIDAR
        msg.angle_min = -self.fov / 2
        msg.angle_max = +self.fov / 2
        msg.angle_increment = self.fov / self.nb_rays
        msg.range_min = 0.05
        msg.range_max = self.lidar.getMaxRange()

        # Copie des valeurs
        msg.ranges = [
            (d if (0 < d < msg.range_max) else float("inf"))
            for d in raw
        ]

        self.publisher.publish(msg)



def main(args=None):
    rclpy.init(args=args) 
    node = TT02DriverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()