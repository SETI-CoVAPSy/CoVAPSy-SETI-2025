#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from controller import Lidar as WebotsLidar
from sensor_msgs.msg import LaserScan
from typing import Literal
from typing_extensions import override

from tt02_driver.gilbert_driver_generic import GilbertDriverGeneric

class TT02DriverNode(Node):
    """ROS 2 driver node controlling the TT-02 in Webots/Simulation."""
    TOPIC_LIDAR: str = "/scan"           # LIDAR topic   (publish)
    TOPIC_COMMAND: str = "/car/command"  # Command topic (subscribe)

    HW_SERIAL_PORT: str = "/dev/ttyTHS1" # Hardware serial port for gilbert STM32
    HW_SERIAL_BAUDRATE: int = 115200     # Hardware serial baudrate
    HW_LIDAR_PORT: str = "/dev/ttyUSB0"  # Hardware LIDAR serial port
    HW_LIDAR_BAUDRATE: int = 256000      # Hardware LIDAR serial baudrate

    def __init__(
            self, 
            target: Literal["hardware", "simulation"],
            node_name: str = "tt02_driver",
            verbose: bool = True
        ) -> None:
        # === Common initialization ===
        super().__init__(node_name)
        self.verbose = verbose
        self._target = target

        self.target_speed = 0.0     # m/s
        self.target_angle = 0.0     # degrees

        # === Common not initialized ===
        self.driver: GilbertDriverGeneric
        self.lidar_nb_rays: int
        self.lidar_fov: float
        self.lidar_range_min: float
        self.lidar_range_max: float

        # === Initialization specific ===
        if target == "hardware":
            self._init_hardware()
        elif target == "simulation":
            self._init_simulation()
        else:
            raise ValueError(f"Unknown target '{target}'")
        
        # === Common initialization final ===
        # subscriber to Ackermann commands
        self.sub_cmd = self.create_subscription(
            AckermannDrive,
            self.TOPIC_COMMAND,
            self.callback_command,
            10
        )
        self.publisher = self.create_publisher(LaserScan, self.TOPIC_LIDAR, 10)
        self.timer = self.create_timer(0.02, self.update)
        self.get_logger().info("TT02 driver started.")

    def _init_hardware(self) -> None:
        from rplidar import RPLidar
        from tt02_driver.gilbert_driver_hardware import GilbertDriverHardware

        self.driver = GilbertDriverHardware(
            serial_port=self.HW_SERIAL_PORT,
            serial_baud=self.HW_SERIAL_BAUDRATE,
            verbose=self.verbose
        )
        
        self.hw_lidar = RPLidar(self.HW_LIDAR_PORT,baudrate=self.HW_LIDAR_BAUDRATE)
        self.hw_lidar.connect()
        self.hw_lidar.start_motor()
        self.hw_lidar_iterator = self.hw_lidar.iter_scans(scan_type='express')
        self.hw_lidar_buffer = np.zeros(360) # Buffer à remplir
    
    def _init_simulation(self) -> None:
        from tt02_driver.gilbert_driver_webots import GilbertDriverWebots
        self.driver = GilbertDriverWebots(verbose=self.verbose)

        self.webots_driver = self.driver.get_driver()
        self.webots_basicTimeStep = int(self.webots_driver.getBasicTimeStep())
        self.sensorTimeStep = 4 * self.webots_basicTimeStep

        self.webots_lidar: WebotsLidar = self.webots_driver.getDevice("RpLidarA2")
        self.webots_lidar.enable(self.sensorTimeStep)

        self.lidar_nb_rays = self.webots_lidar.getHorizontalResolution()
        self.lidar_fov = self.webots_lidar.getFov()
        self.lidar_range_min = 0.05  # Minimum range in meters
        self.lidar_range_max = self.webots_lidar.getMaxRange()

    def callback_command(self, msg: AckermannDrive) -> None:
        """Callback from command topic."""
        self.target_speed = msg.speed
        self.target_angle = msg.steering_angle * 180.0 / 3.14159265358979
        self.target_speed = max(self.driver.SPEED_LIMIT_REVERSE, min(self.driver.SPEED_LIMIT_FORWARD, self.target_speed))
        self.target_angle = max(-self.driver.ANGLE_LIMIT_DEG, min(self.driver.ANGLE_LIMIT_DEG, self.target_angle))
        
    def update(self) -> None:
        """Clock update function called periodically."""
        if self._target == "hardware":
            pass # Nothing to do yet
        elif self._target == "simulation":
            step_result = self.webots_driver.step()

            if step_result == -1:
                # Webots simulation ended (window closed)
                self.get_logger().warn("Simulation ended.")
                rclpy.shutdown()
                return
            
        self.driver.set_speed_mps(self.target_speed)
        self.driver.set_steering_angle_deg(self.target_angle)
        self.publish_scan()

    def publish_scan(self) -> None:
        """Publishes last LIDAR scan."""
        # Création du message
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "RpLidarA2"

        # Champs LIDAR
        msg.angle_min = -self.lidar_fov / 2
        msg.angle_max = +self.lidar_fov / 2
        msg.angle_increment = self.lidar_fov / self.lidar_nb_rays
        msg.range_min = self.lidar_range_min
        msg.range_max = self.lidar_range_max

        # Ranges
        if self._target == "simulation":
            raw = self.webots_lidar.getRangeImage()
            msg.ranges = [ # Copie des valeurs
                (d if (0 <= d < msg.range_max) else float("inf"))
                for d in raw
            ]
        elif self._target == "hardware":
            lidar_scan = next(self.hw_lidar_iterator)
            for i in range(len(lidar_scan)):
                angle = min(359,max(0,359-int(lidar_scan[i][1])))
                self.hw_lidar_buffer[angle] = lidar_scan[i][2]
            msg.ranges = [ # Copie des valeurs
                (d if (0 <= d < msg.range_max) else float("inf"))
                for d in self.hw_lidar_buffer
            ]

        self.publisher.publish(msg)
    
    @override
    def destroy_node(self) -> None:
        self.driver.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args) 
    node = TT02DriverNode("simulation")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()