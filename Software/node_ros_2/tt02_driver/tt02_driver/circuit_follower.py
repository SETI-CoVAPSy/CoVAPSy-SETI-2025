import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
import numpy as np

class CircuitFollower(Node):
    def __init__(self):
        super().__init__('circuit_follower')

        # S'abonne au topic publié par ton driver_node.py
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)

        # Publie sur le topic écouté par ton driver_node.py
        self.publisher_ = self.create_publisher(AckermannDrive, '/car/command', 10)

        # Paramètres de conduite
        self.target_speed = 0.8  # Vitesse prudente pour le SLAM
        self.safe_dist = 0.6     # Distance souhaitée par rapport aux murs (mètres)

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
        num_points = len(ranges)

        # 1. Définition des zones (Indexation pour un LiDAR 360°)
        # Devant : +/- 20 degrés autour de l'index central
        front_indices = range(int(num_points*0.45), int(num_points*0.55))
        dist_front = np.min(ranges[front_indices])

        # Côtés pour le maintien au centre
        dist_right = np.mean(ranges[int(num_points*0.2):int(num_points*0.3)])
        dist_left = np.mean(ranges[int(num_points*0.7):int(num_points*0.8)])

        drive_msg = AckermannDrive()

        # 2. LOGIQUE D'ÉVITEMENT
        if dist_front < 1.0: # Si obstacle à moins d'un mètre
            self.get_logger().info(f"Obstacle à {dist_front:.2f}m : Tentative de contournement")

            # On ralentit mais on ne s'arrête pas
            drive_msg.speed = 0.5

            # On choisit de tourner du côté où il y a le plus de place
            if dist_left > dist_right:
                drive_msg.steering_angle = 0.6  # Tourne à gauche (positif)
            else:
                drive_msg.steering_angle = -0.6 # Tourne à droite (négatif)

        else:
            # 3. LOGIQUE DE CROISIÈRE (Maintien au centre)
            drive_msg.speed = self.target_speed
            error = dist_left - dist_right
            # On applique un gain pour la direction (à ajuster selon la largeur de la piste)
            drive_msg.steering_angle = error * 0.3

        # 4. Limites de sécurité pour le servo de direction (en radians)
        # Ton driver convertit en degrés après, mais restons prudents
        drive_msg.steering_angle = max(min(drive_msg.steering_angle, 0.7), -0.7)

        self.publisher_.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CircuitFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
