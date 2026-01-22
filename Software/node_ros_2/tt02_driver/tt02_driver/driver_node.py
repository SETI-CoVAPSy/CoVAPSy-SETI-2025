"""
Base ROS 2 driver node for TT-02 in Webots/Simulation or Hardware.
"""

import math
import rclpy
import numpy as np
from rclpy.time import Time
from rclpy.node import Node
from controller import Lidar as WebotsLidar
from typing import Literal
from typing_extensions import override

# ROS 2 Message imports
from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


#TF imports
import tf2_ros

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

        # === Odometry initialization ===
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = self.get_clock().now()
        self.odom_publisher = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.path_pub = self.create_publisher(Path, '/path', 10)
        self.k = 0.9

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

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
        #self.timer = self.create_timer(0.02, self.update)
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

        self.sim_time = self.webots_driver.getTime()

        self.webots_lidar: WebotsLidar = self.webots_driver.getDevice("RpLidarA2")
        self.webots_lidar.enable(self.sensorTimeStep)

        self.lidar_nb_rays = self.webots_lidar.getHorizontalResolution()
        self.lidar_fov = self.webots_lidar.getFov()
        self.lidar_range_min = 0.05  # Minimum range in meters
        self.lidar_range_max = self.webots_lidar.getMaxRange()
        self.angle_min = -math.pi
        self.angle_max = math.pi
        
        self.clock_pub = self.create_publisher(Clock, '/clock', 10) #sim clock publisher init

    def callback_command(self, msg: AckermannDrive) -> None:
        """Callback from command topic."""
        self.target_speed = msg.speed
        self.target_angle = msg.steering_angle * 180.0 / 3.14159265358979
        self.target_speed = max(self.driver.SPEED_LIMIT_REVERSE, min(self.driver.SPEED_LIMIT_FORWARD, self.target_speed))
        self.target_angle = max(-self.driver.ANGLE_LIMIT_DEG, min(self.driver.ANGLE_LIMIT_DEG, self.target_angle))
        
    def update(self) -> None:
        """Clock update function called periodically."""
        current_time = self.get_clock().now()
        if self._target == "hardware":
            dt = (current_time - self.last_time).nanoseconds / 1e9
        elif self._target == "simulation":
            dt = self.webots_basicTimeStep * 1e-3
            step_result = self.webots_driver.step()

            if step_result == -1:
                # Webots simulation ended (window closed)
                self.get_logger().warn("Simulation ended.")
                rclpy.shutdown()
                return

        
        # --- Odometry Calculation (Dead Reckoning) ---
        v = self.target_speed
        alpha = self.target_angle * math.pi / 180.0
        L = 0.257  # from tt02 proto file

        # Ackermann kinematic model
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += self.k * (v / L) * math.tan(alpha) * dt

        self.publish_odometry(current_time)
        self.last_time = current_time
            
        self.driver.set_speed_mps(self.target_speed)
        self.driver.set_steering_angle_deg(self.target_angle)
        self.publish_scan()

        self.publish_clock()

    def publish_odometry(self, time: Time) -> None:
        # 1. Broadcast Transform
        t = TransformStamped()
        t.header.stamp = time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation = self.yaw_to_quaternion(self.theta)
        self.tf_broadcaster.sendTransform(t)

        # 2. Publish Odometry Message
        odom = Odometry()
        odom.header.stamp = time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = self.yaw_to_quaternion(self.theta)
        odom.twist.twist.linear.x = self.target_speed
        self.odom_publisher.publish(odom)

        # 3. Publisher Odometry Path
        ps = PoseStamped()
        ps.header.stamp = time.to_msg()
        ps.header.frame_id = 'odom'
        ps.pose = odom.pose.pose

        self.path_msg.header.stamp = time.to_msg()
        self.path_msg.poses.append(ps)
        self.path_pub.publish(self.path_msg)

    def yaw_to_quaternion(self, yaw: float) -> Quaternion:

        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q


    def publish_clock(self) -> None:
        sim_time = self.webots_driver.getTime() 

        clock_msg = Clock()
        clock_msg.clock.sec = int(sim_time)
        clock_msg.clock.nanosec = int((sim_time - int(sim_time)) * 1e9)

        self.clock_pub.publish(clock_msg)

    def publish_scan(self) -> None:
        """Publishes last LIDAR scan."""
        # Création du message
        msg = LaserScan()   
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "RpLidarA2"

        # Champs LIDAR
        msg.angle_min = self.angle_min
        msg.angle_max = self.angle_max
        msg.range_min = self.lidar_range_min
        msg.range_max = self.lidar_range_max

        # Ranges
        if self._target == "simulation":
            raw = list(self.webots_lidar.getRangeImage())
            N = len(raw)
            msg.angle_increment = (msg.angle_max - msg.angle_min) / (N - 1)
            msg.ranges = [
                (d if (msg.range_min <= d <= msg.range_max) else float("inf"))
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
    
    def run(self):
        while rclpy.ok():
            self.update()
            rclpy.spin_once(self, timeout_sec=0)

    @override
    def destroy_node(self) -> None:
        self.driver.close()
        super().destroy_node()

def main(args: list[str] | None = None):
    rclpy.init(args=args) 
    node = TT02DriverNode("simulation")
    #rclpy.spin(node)
    try:
        node.run()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
