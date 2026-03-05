#!/usr/bin/env python3
from __future__ import division

import os
import sys
import csv
import math
import time
import importlib
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import LaserScan

#cette merde a été faite par IA pour retrouver le solver compilé en Rust généré par OpEn, qui n'est pas un package Python classique 
# et peut être dans différents dossiers selon les machines et les configurations. 

# à supprimer
def _resolve_optimizer_path():
    """Résout dynamiquement le dossier contenant le module solver compilé.

    Ordre de recherche:
    1) variable d'env MPC_OPTIMIZER_PATH
    2) chemins candidats autour du workspace/fichier courant
    """
    checked = []

    def _is_valid_optimizer_dir(path_obj: Path):
        if not path_obj.exists() or not path_obj.is_dir():
            return False
        module_so = path_obj / "dynamic_racing_target_point.so"
        return module_so.exists()

    env_path = os.getenv("MPC_OPTIMIZER_PATH")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        checked.append(str(candidate))
        if _is_valid_optimizer_dir(candidate):
            return str(candidate)

    repo_name = "A-Nonlinear-Model-Predictive-Control-Strategy-for-Autonomous-Racing-of-Scale-Vehicles"

    roots = []
    if os.getenv("MPC_REPO_ROOT"):
        roots.append(Path(os.getenv("MPC_REPO_ROOT")).expanduser().resolve())
    roots.append(Path.cwd().resolve())
    roots.extend(Path.cwd().resolve().parents)
    roots.extend(Path(__file__).resolve().parents)

    # keep order but remove duplicates
    unique_roots = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)

    for root in unique_roots:
        candidates = [
            root / repo_name / "PANOC_DYNAMIC_MOTOR_MODEL" / "dynamic_my_optimizer" / "dynamic_racing_target_point",
            root / "PANOC_DYNAMIC_MOTOR_MODEL" / "dynamic_my_optimizer" / "dynamic_racing_target_point",
        ]
        for candidate in candidates:
            checked.append(str(candidate))
            if _is_valid_optimizer_dir(candidate):
                return str(candidate)

    raise RuntimeError(
        "Impossible de localiser le build MPC (dynamic_racing_target_point.so). "
        "Définis MPC_OPTIMIZER_PATH vers .../PANOC_DYNAMIC_MOTOR_MODEL/dynamic_my_optimizer/dynamic_racing_target_point. "
        f"Chemins testés: {checked[:10]}{' ...' if len(checked) > 10 else ''}"
    )


def _get_bool_env(name, default=False):
    """Lit un booléen depuis une variable d'environnement."""
    value = os.getenv(name, str(int(default))).strip().lower()
    return value in ("1", "true", "yes", "on")


def euler_from_quaternion(x, y, z, w):
    """Conversion quaternion -> angles d'Euler (roll, pitch, yaw)."""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


# Note: N/n_states/n_controls doivent rester cohérents avec le solver généré OpEn.
N = 50 # horizon MPC (nombre de points projetés)
n_states = 6 # [x, y, yaw, vx_body, vy_body, yaw_rate]
n_controls = 2 # [accélération longitudinale, angle de braquage]



ODOM_TOPIC = os.getenv("MPC_ODOM_TOPIC", "/odom")
WAYPOINTS_TOPIC = os.getenv("MPC_WAYPOINTS_TOPIC", "/path")
SCAN_TOPIC = os.getenv("MPC_SCAN_TOPIC", "/scan")
CMD_TOPIC = os.getenv("MPC_CMD_TOPIC", "/car/command")

#source de référence pour le MPC: waypoints (si activés) ou LiDAR (sinon)
# Si USE_WAYPOINTS=True, on suit /path.
# Sinon, on construit une référence depuis le LiDAR (/scan).
USE_WAYPOINTS = _get_bool_env("MPC_USE_WAYPOINTS", False)


# à définir: nombre de points projetés pour construire la trajectoire de référence à partir des waypoints
PATH_LOOKAHEAD_POINTS = int(os.getenv("MPC_PATH_LOOKAHEAD_POINTS", "5"))

# à définir: nombre de points entre les waypoints projetés pour construire la trajectoire de référence (si waypoints)
PATH_STEP_POINTS = int(os.getenv("MPC_PATH_STEP_POINTS", "3"))

# à définir: est-ce que la trajectoire de référence construite à partir des waypoints doit être cyclique (True) ou saturée aux extrémités (False)
PATH_IS_CYCLIC = _get_bool_env("MPC_PATH_IS_CYCLIC", True)

# à définir: distance de projection LiDAR vers l'avant (m)
LIDAR_LOOKAHEAD_M = float(os.getenv("MPC_LIDAR_LOOKAHEAD_M", "1.0"))
# à définir: pas de discrétisation de la référence LiDAR (m)
LIDAR_STEP_M = float(os.getenv("MPC_LIDAR_STEP_M", "0.20"))
# à définir: distance mini devant la voiture pour garder les points LiDAR (m)
LIDAR_MIN_FORWARD_X_M = float(os.getenv("MPC_LIDAR_MIN_FORWARD_X_M", "0.05"))
# à définir: nombre mini de points par côté (gauche/droite) pour estimer le centre
LIDAR_MIN_POINTS_PER_SIDE = int(os.getenv("MPC_LIDAR_MIN_POINTS_PER_SIDE", "3"))

# à définir: vitesse longitudinale minimale utilisée par le modèle MPC (évite vx≈0)
MPC_MIN_ABS_VX = float(os.getenv("MPC_MIN_ABS_VX", "0.5"))
# à définir: fallback de référence minimale (ligne droite) si pas de waypoints et LiDAR insuffisant
FALLBACK_STEP_M = float(os.getenv("MPC_FALLBACK_STEP_M", "0.20"))
# à définir: lookahead de référence pour fallback ligne droite (m)
FALLBACK_LOOKAHEAD_M = float(os.getenv("MPC_FALLBACK_LOOKAHEAD_M", "0.2"))
# à définir: gain de conversion entre la commande MPC (u_cl1) et la vitesse linéaire du véhicule
ACKERMANN_SPEED_SCALE = float(os.getenv("MPC_ACKERMANN_SPEED_SCALE", "1.0"))
# à définir: vitesse de sécurité envoyée en commande de secours si le solveur échoue
MPC_FAILSAFE_SPEED = float(os.getenv("MPC_FAILSAFE_SPEED", "0.0"))

LOG_PATH = Path(os.getenv("MPC_LOG_PATH", str(Path.cwd() / "race_DATA.csv"))).expanduser().resolve()
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

#Solver compilé en Rust et exposé via un binding Python généré par OpEn.
sys.path.insert(1, _resolve_optimizer_path())
dynamic_racing_target_point = importlib.import_module("dynamic_racing_target_point")


class MPCNode(Node):
    """Node ROS2 qui convertit odométrie + perception Lidar en commandes Ackermann via MPC."""

    def __init__(self):
        super().__init__("mpc_node")

        # Instance du solveur MPC
        self.solver = dynamic_racing_target_point.solver()
        self.waypoints = np.empty((0, 2), dtype=float)
        self.latest_scan = None

        
        self.log_handle = open(str(LOG_PATH), "w", newline="")
        self.writer = csv.writer(self.log_handle)

        self.pub_cmd = self.create_publisher(AckermannDrive, CMD_TOPIC, 10)
        self.sub_odom = self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)

        self.sub_waypoints = None
        if USE_WAYPOINTS:
            self.sub_waypoints = self.create_subscription(NavPath, WAYPOINTS_TOPIC, self.waypoints_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_callback, 10)

        self.mpciter = 0
        self.u_cl1 = 0.0
        self.u_cl2 = 0.0

        # Warm-start du solveur: on réinjecte la dernière séquence optimale
        self.guess = [0.0] * (2 * N)
        self.theta2unwrap = []

        # x0 = [x, y, yaw, vx_body, vy_body, yaw_rate]
        self.x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_valid_vx = 0.0
        self.last_lidar_center_y = None
        self.min_front_dist = 10.0 #

        self.get_logger().info(
            "MPC ROS2 started | odom=%s cmd=%s use_waypoints=%s (sinon LiDAR)"
            % (ODOM_TOPIC, CMD_TOPIC, str(USE_WAYPOINTS))
        )
        

    def _publish_failsafe_cmd(self):
        """Commande de sécurité envoyée quand le solveur échoue."""
        cmd = AckermannDrive()
        cmd.speed = MPC_FAILSAFE_SPEED
        cmd.steering_angle = 0.0
        self.pub_cmd.publish(cmd)

    def diagnostic_check(self):
        cmd_subscribers = self.pub_cmd.get_subscription_count()
        if cmd_subscribers == 0:
            self.get_logger().warning(
                "Aucun subscriber sur %s. Lance le node driver (tt02_driver) et Webots." % CMD_TOPIC
            )

        odom_publishers = self.count_publishers(ODOM_TOPIC)
        if odom_publishers == 0:
            self.get_logger().warning(
                "Aucun publisher sur %s. Le MPC ne peut pas tourner sans odométrie." % ODOM_TOPIC
            )

        if self.count_publishers(SCAN_TOPIC) == 0:
            self.get_logger().warning(
                "Aucun publisher sur %s. Référence LiDAR indisponible, fallback ligne droite." % SCAN_TOPIC
            )

        if USE_WAYPOINTS and self.count_publishers(WAYPOINTS_TOPIC) == 0:
            self.get_logger().info(
                "Aucun publisher sur %s. Le MPC utilisera le LiDAR." % WAYPOINTS_TOPIC
            )

    def destroy_node(self):
        """Destructeur du node: ferme le fichier de log proprement."""
        try:
            self.log_handle.close()
        except Exception:
            pass
        return super().destroy_node()

    #+================================+
    #| Trajectoire optimisée calculée |
    #+================================+
    def waypoints_callback(self, msg):
        """Met à jour la trajectoire de référence reçue sur /path."""
        if len(msg.poses) == 0:
            return
        #stockage des waypoints reçus dans un tableau
        self.waypoints = np.array([(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses], dtype=float)

    def scan_callback(self, msg):
        """Mémorise le dernier scan LiDAR."""
        self.latest_scan = msg

    def _idx_path(self, idx, size):
        """Gère les indices pour la trajectoire de référence."""
        
        if PATH_IS_CYCLIC:
            return idx % size #0 à size-1 en boucle
        if idx < 0:
            return 0
        if idx >= size:
            return size - 1
        return idx

    def _reference_from_waypoints(self, x_odom, y_odom):
        """Construit cible lointaine + trajectoire locale 
        à partir de la trajectoire de référence."""

        #si on a moins de 2 waypoints, on peut pas poursuivre.
        if self.waypoints.shape[0] < 2:
            return None #todo voir comment gérer ça proprement

        #on trouve l'indice du waypoint le plus proche de la position courante
        diff = self.waypoints - np.array([x_odom, y_odom])
        nearest_idx = int(np.argmin(np.sum(diff ** 2, axis=1)))
        size = self.waypoints.shape[0]

        #on choisit la cible lointaine à un certain nombre de points après le plus proche
        target_idx = self._idx_path(nearest_idx + PATH_LOOKAHEAD_POINTS, size)
        target_x = self.waypoints[target_idx, 0]
        target_y = self.waypoints[target_idx, 1]

        # on construit la trajectoire projetée à partir de plusieurs points espacés sur la trajectoire de référence
        proj_center_X = np.zeros(N)
        proj_center_Y = np.zeros(N)
        for i in range(N):
            #Des mini waypoints sont créés entre les waypoints de la trajectoire
            idx = self._idx_path(nearest_idx + (i + 1) * PATH_STEP_POINTS, size)
            proj_center_X[i] = self.waypoints[idx, 0]
            proj_center_Y[i] = self.waypoints[idx, 1]

        return target_x, target_y, proj_center_X, proj_center_Y, "waypoints"

    #+================================+
    #| FALLBACK SUR LE LIDAR EN LOCAL |
    #+================================+
    def _scan_to_points_xy(self):
        """Convertit LaserScan en points XY dans le repère capteur (plan 2D)."""
        if self.latest_scan is None:
            return None #todo passer au plan C -> pas de lidar
        if len(self.latest_scan.ranges) == 0:
            return None

        # Convertit chaque mesure valide en point (x, y) du repère capteur.
        pts = []
        angle = self.latest_scan.angle_min
        for d in self.latest_scan.ranges:

            # Ignore les distances invalides ou hors plage du LiDAR.
            if math.isfinite(d) and self.latest_scan.range_min <= d <= self.latest_scan.range_max:
                x = d * math.cos(angle)
                y = d * math.sin(angle)
                pts.append((x, y))
            angle += self.latest_scan.angle_increment

        if len(pts) == 0:
            return None
        
        # Retourne les points LiDAR dans le repère local.
        return np.array(pts, dtype=float)

    def _build_reference_from_center_offset(self, x_odom, y_odom, theta_odom, center_y_local, lookahead, source_name):
        """Construit la cible et la trajectoire projetée à partir d'un offset latéral du centre estimé par le LiDAR."""
        # Base locale: projection selon le cap du véhicule.
        ct = math.cos(theta_odom)
        st = math.sin(theta_odom)

        # Cible en avance, décalée latéralement par center_y_local.
        target_x = x_odom + lookahead * ct - center_y_local * st
        target_y = y_odom + lookahead * st + center_y_local * ct

        proj_center_X = np.zeros(N)
        proj_center_Y = np.zeros(N)
        
        # Echantillonne N points entre la pose courante et la cible.
        step = lookahead / N if lookahead > 0 else 0.0
        
        for i in range(N):
            s = (i + 1) * step
            proj_center_X[i] = x_odom + s * ct - center_y_local * st
            proj_center_Y[i] = y_odom + s * st + center_y_local * ct

        return target_x, target_y, proj_center_X, proj_center_Y, source_name

    def _reference_from_lidar(self, x_odom, y_odom, theta_odom):
        """Construit cible + trajectoire locale à partir des points LiDAR filtrés et projetés vers l'avant."""
        points = self._scan_to_points_xy()
        if points is None:
            return None

        # Garde uniquement les points devant la voiture.
        front_mask = (points[:, 0] > LIDAR_MIN_FORWARD_X_M) & (points[:, 0] < 5.0)
        forward_points = points[front_mask]

        if forward_points.shape[0] == 0:
            return None

        # Estime la distance frontale minimale pour le freinage de sécurité.
        center_mask = np.abs(forward_points[:, 1]) < 0.3
        center_points = forward_points[center_mask]
        
        if center_points.shape[0] > 0:
            self.min_front_dist = float(np.min(center_points[:, 0]))

        # Sélectionne le point le plus éloigné comme direction locale.
        distances = np.linalg.norm(forward_points, axis=1)
        max_idx = np.argmax(distances)
        best_point = forward_points[max_idx]

        # Ajuste la projection selon l'espace libre disponible.
        dynamic_lookahead = max(0.5, min(LIDAR_LOOKAHEAD_M, self.min_front_dist - 0.2))

        # Convertit l'angle du point choisi en décalage latéral cible.
        angle = math.atan2(best_point[1], best_point[0])
        target_y_local = dynamic_lookahead * math.sin(angle)
        
        self.last_lidar_center_y = target_y_local

        return self._build_reference_from_center_offset(
            x_odom, y_odom, theta_odom, target_y_local, dynamic_lookahead, "lidar_points"
        )

    def _reference_straight_fallback(self, x_odom, y_odom, theta_odom):
        """Fallback minimal si pas de waypoints et LiDAR insuffisant."""
        ct = math.cos(theta_odom)
        st = math.sin(theta_odom)

        # Plan C: trajectoire droite courte devant le véhicule.
        target_x = x_odom + FALLBACK_LOOKAHEAD_M * ct
        target_y = y_odom + FALLBACK_LOOKAHEAD_M * st

        proj_center_X = np.zeros(N)
        proj_center_Y = np.zeros(N)
        for i in range(N):
            s = (i + 1) * FALLBACK_STEP_M
            proj_center_X[i] = x_odom + s * ct
            proj_center_Y[i] = y_odom + s * st

        return target_x, target_y, proj_center_X, proj_center_Y, "straight"

    def _sanitize_vx_for_solver(self, vx_candidate):
        """Stabilise vx pour éviter les valeurs infinies qui nuisent au solveur."""
        # Si mesure invalide, conserve la dernière vitesse exploitable.
        if not math.isfinite(vx_candidate):
            return self.last_valid_vx

        # Evite les vitesses trop proches de 0 pour la stabilité numérique.
        if abs(vx_candidate) < MPC_MIN_ABS_VX:
            if abs(self.last_valid_vx) >= MPC_MIN_ABS_VX:
                return self.last_valid_vx
            sign = 1.0 if vx_candidate >= 0.0 else -1.0
            return sign * MPC_MIN_ABS_VX

        self.last_valid_vx = vx_candidate
        return vx_candidate

    def callback(self, data):
        """Boucle MPC principale appelée à chaque message odométrie."""
        pose = data.pose.pose
        euler = euler_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        theta_odom = euler[2]

        # Unwrap de yaw pour éviter des sauts ±pi qui perturbent le solveur
        self.theta2unwrap.append(theta_odom)
        thetaunwrapped = np.unwrap(self.theta2unwrap)

        x_odom = pose.position.x
        y_odom = pose.position.y

        twist = data.twist.twist

        # Construction de l'état courant x0 utilisé par le solveur
        self.x0[0] = x_odom
        self.x0[1] = y_odom
        self.x0[2] = thetaunwrapped[-1]
        if self.mpciter < 15:
            self.x0[3] = 3.0
        else:
            vx_body = twist.linear.x * math.cos(self.x0[2]) + twist.linear.y * math.sin(self.x0[2])
            self.x0[3] = self._sanitize_vx_for_solver(vx_body)
        self.x0[4] = twist.linear.y * math.cos(self.x0[2]) - twist.linear.x * math.sin(self.x0[2])
        self.x0[5] = twist.angular.z

        reference = None

        # Priorité 1: waypoints (si activés et disponibles)
        if USE_WAYPOINTS:
            reference = self._reference_from_waypoints(self.x0[0], self.x0[1])

        # Priorité 2: LiDAR (navigation locale)
        if reference is None:
            reference = self._reference_from_lidar(self.x0[0], self.x0[1], self.x0[2])

        # Priorité 3: si perception indisponible ponctuellement, on garde le dernier offset LiDAR
        # mais on recalcule la projection avec la pose courante (pas de référence figée)
        if reference is None and self.last_lidar_center_y is not None:
            reference = self._build_reference_from_center_offset(
                self.x0[0],
                self.x0[1],
                self.x0[2],
                self.last_lidar_center_y,
                LIDAR_LOOKAHEAD_M,
                "lidar_points_hold",
            )

        # Priorité 4: ligne droite (sécurité) uniquement si aucune référence n'a jamais été obtenue
        if reference is None:
            reference = self._reference_straight_fallback(self.x0[0], self.x0[1], self.x0[2])

        target_x, target_y, proj_center_X, proj_center_Y, source = reference

        parameter = []
        for i in range(n_states):
            parameter.append(self.x0[i])
        parameter.append(self.u_cl1)
        parameter.append(self.u_cl2)
        parameter.append(target_x)
        parameter.append(target_y)
        for i in range(N):
            parameter.append(proj_center_X[i])
            parameter.append(proj_center_Y[i])

        parameter_np = np.array(parameter, dtype=float)
        if not np.all(np.isfinite(parameter_np)):
            self.get_logger().warning("Paramètres MPC non finis, commande failsafe publiée")
            self._publish_failsafe_cmd()
            return

        try:
            # Appel solveur MPC (warm-start via self.guess)
            t_start = time.time()
            result = self.solver.run(
                p=parameter_np.tolist(),
                initial_guess=[self.guess[i] for i in range(n_controls * N)],
            )
            elapsed = time.time() - t_start
        except Exception as exc:
            self.get_logger().error(f"Erreur solver MPC: {exc}")
            self._publish_failsafe_cmd()
            return

        solution = getattr(result, "solution", None)
        if solution is None:
            # Le binding OpEn retourne None en cas d'échec numérique (ex: code 2000)
            self.get_logger().warning("Solver MPC sans solution exploitable, commande failsafe publiée")
            self._publish_failsafe_cmd()
            return

        u_star = np.array(solution, dtype=float)
        if u_star.shape[0] < 2:
            self.get_logger().warning("Solution MPC invalide")
            self._publish_failsafe_cmd()
            return
        if not np.isfinite(u_star[0]) or not np.isfinite(u_star[1]):
            self.get_logger().warning("Solution MPC non-finie, commande failsafe publiée")
            self._publish_failsafe_cmd()
            return

        self.guess = u_star.tolist()
        self.u_cl1 = float(u_star[0])
        self.u_cl2 = float(u_star[1])

        cmd = AckermannDrive()
        # Conversion vers la commande véhicule
        cmd = AckermannDrive()
        cmd.speed = self.u_cl1 * ACKERMANN_SPEED_SCALE
        cmd.steering_angle = self.u_cl2

        # override AEB based on LiDAR obstacle
        if self.min_front_dist < 0.6:
            cmd.speed = 0.0
        elif self.min_front_dist < 1.2:
            cmd.speed *= (self.min_front_dist - 0.6) / 0.6

        self.pub_cmd.publish(cmd)

        if self.mpciter % 30 == 0:
            self.get_logger().info(
                "MPC source=%s speed=%.3f steer=%.3f" % (source, cmd.speed, cmd.steering_angle)
            )

        stamp = data.header.stamp
        row = [
            self.x0[0],
            self.x0[1],
            self.x0[2],
            self.x0[3],
            self.x0[4],
            self.x0[5],
            elapsed,
            self.u_cl1,
            self.u_cl2,
            source,
            stamp.sec,
            stamp.nanosec,
        ]
        self.writer.writerow(row)
        self.log_handle.flush()

        self.mpciter += 1


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
