"""ROS2 MPC node publishing Ackermann commands for the TT02 driver."""

from __future__ import annotations

import math
import importlib
import traceback
from dataclasses import dataclass

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from rclpy.node import Node

from tt02_driver.gilbert_driver_generic import GilbertDriverGeneric

from .MPC import MPC
from .spatial_bicycle_models import BicycleModel, TemporalState

try:
    sparse = importlib.import_module("scipy.sparse")
except ModuleNotFoundError:
    sparse = None


@dataclass
class MPCWaypoint:
    """Waypoint container used by the MPC controller."""

    x: float
    y: float
    psi: float
    kappa: float
    v_ref: float
    lb: float
    ub: float
    static_border_cells: tuple[None, None] = (None, None)
    dynamic_border_cells: tuple[None, None] = (None, None)

    def __sub__(self, other: "MPCWaypoint") -> float:
        return float(math.hypot(self.x - other.x, self.y - other.y))


class RosReferencePath:
    """Minimal reference path interface expected by the legacy MPC code."""

    def __init__(
        self,
        wp_x: list[float],
        wp_y: list[float],
        v_ref: float,
        wp_lb: list[float] | None = None,
        wp_ub: list[float] | None = None,
        circular: bool = True,
    ) -> None:
        if len(wp_x) != len(wp_y):
            raise ValueError("waypoints_x and waypoints_y must have same length")
        if len(wp_x) < 3:
            raise ValueError("at least 3 waypoints are required")
        if wp_lb is not None and len(wp_lb) != len(wp_x):
            raise ValueError("waypoints_lb must match waypoints_x length")
        if wp_ub is not None and len(wp_ub) != len(wp_x):
            raise ValueError("waypoints_ub must match waypoints_x length")

        self.circular = circular
        self._eps = 1e-9

        waypoints_xy = list(zip(wp_x, wp_y))
        self.waypoints = self._build_waypoints(waypoints_xy, v_ref, wp_lb, wp_ub)
        self.n_waypoints = len(self.waypoints)
        self.length, self.segment_lengths = self._compute_length()
        self.cumulative_lengths = np.cumsum(self.segment_lengths)

    def _wrap_id(self, wp_id: int) -> int:
        if self.circular:
            return wp_id % self.n_waypoints
        return int(np.clip(wp_id, 0, self.n_waypoints - 1))

    def _build_waypoints(
        self,
        points: list[tuple[float, float]],
        v_ref: float,
        wp_lb: list[float] | None,
        wp_ub: list[float] | None,
    ) -> list[MPCWaypoint]:
        waypoints: list[MPCWaypoint] = []
        n_points = len(points)

        for i in range(n_points):
            x, y = points[i]
            prev_i = (i - 1) % n_points if self.circular else max(0, i - 1)
            next_i = (i + 1) % n_points if self.circular else min(n_points - 1, i + 1)

            prev_x, prev_y = points[prev_i]
            next_x, next_y = points[next_i]

            heading = math.atan2(next_y - y, next_x - x)
            seg_len = max(math.hypot(next_x - x, next_y - y), self._eps)

            if i == 0 and not self.circular:
                kappa = 0.0
            else:
                prev_heading = math.atan2(y - prev_y, x - prev_x)
                angle_diff = (heading - prev_heading + math.pi) % (2.0 * math.pi) - math.pi
                kappa = angle_diff / seg_len

            if wp_lb is not None and wp_ub is not None:
                lb_val = float(wp_lb[i])
                ub_val = float(wp_ub[i])
                if ub_val < lb_val:
                    lb_val, ub_val = ub_val, lb_val
            else:
                # Adaptive fallback when no lateral bounds are provided:
                # Use a fixed default corridor width (e.g. 1.0m half-width -> 2m track)
                # This avoids collapsing the track if waypoints are dense.
                lb_val = -1.0
                ub_val = 1.0

            waypoints.append(
                MPCWaypoint(
                    x=float(x),
                    y=float(y),
                    psi=float(heading),
                    kappa=float(kappa),
                    v_ref=float(v_ref),
                    lb=lb_val,
                    ub=ub_val,
                )
            )

        return waypoints

    def _compute_length(self) -> tuple[float, list[float]]:
        segment_lengths = [0.0]
        for i in range(self.n_waypoints - 1):
            segment_lengths.append(self.waypoints[i + 1] - self.waypoints[i])

        if self.circular:
            segment_lengths[0] = self.waypoints[0] - self.waypoints[-1]

        total = float(sum(segment_lengths))
        return total, segment_lengths

    def get_waypoint(self, wp_id: int) -> MPCWaypoint:
        return self.waypoints[self._wrap_id(wp_id)]

    def update_path_constraints(
        self,
        start_wp_id: int,
        horizon: int,
        left_margin: float,
        right_margin: float,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        ub = np.zeros(horizon)
        lb = np.zeros(horizon)

        for i in range(horizon):
            wp = self.get_waypoint(start_wp_id + i)
            ub_i = wp.ub - left_margin
            lb_i = wp.lb + right_margin

            ub[i] = ub_i
            lb[i] = lb_i

        return ub, lb, []


class TT02MPCNode(Node):
    """ROS2 MPC node using TT02 driver topics and Ackermann commands."""

    TOPIC_ODOM = "/odom"
    TOPIC_COMMAND = "/car/command"

    def __init__(self) -> None:
        super().__init__("tt02_mpc")

        if sparse is None:
            raise RuntimeError(
                "scipy is required for MPC node. Install with `pip install scipy osqp`."
            )

        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("horizon", 20)
        self.declare_parameter("reference_speed", 0.8)
        self.declare_parameter("ay_max", 3.0)
        self.declare_parameter("max_steering_angle_deg", GilbertDriverGeneric.ANGLE_LIMIT_DEG)
        self.declare_parameter("q_ey", 1.0)
        self.declare_parameter("q_epsi", 0.7)
        self.declare_parameter("q_t", 0.0)
        self.declare_parameter("qn_ey", 1.0)
        self.declare_parameter("qn_epsi", 1.0)
        self.declare_parameter("qn_t", 0.0)
        self.declare_parameter("r_speed", 0.5)
        self.declare_parameter("r_steer", 0.05)
        self.declare_parameter("osqp_max_iter", 20000)
        self.declare_parameter("osqp_eps_abs", 1e-3)
        self.declare_parameter("osqp_eps_rel", 1e-3)
        self.declare_parameter("wp_ordered_mode", True)
        # -1 selects nearest waypoint at startup, then keeps strict ordered
        # traversal from that point.
        self.declare_parameter("wp_ordered_start_index", -1)
        self.declare_parameter("wp_reached_distance", 0.45)
        self.declare_parameter("wp_pass_margin", 0.05)
        self.declare_parameter("debug_decisions", False)
        self.declare_parameter("debug_decisions_every", 10)

        # Example waypoints for a simple track (can be overridden by parameters)
        # from test_track.png
        # self.declare_parameter(
        #     "waypoints_x",
        #     [-4.3, -4.3, -1.6, -0.58, 0.6, -0.4, -0.4, 0.86, 2.6, 3.61, 2.63, 3.55, 3.55, 1.6, 5.55, 5.55, -0.28, -2.27, -2.54],
        # )
        # self.declare_parameter(
        #     "waypoints_y",
        #     [-1.6, -4.3, -4.3, -3.44, -1.5, -0.2, 1.3, 2.40, 3.53, 2.54, 0.51, -1.93, -3.53, -4.5, -5.45, 5.37, 5.37, 3.41, -1.25],
        # )

        #more affine waypoints

        self.declare_parameter(
            "waypoints_x",
            [-3.67578, -4.25027, -4.00572, -2.98572, -1.75572, -0.71572, 0.19428, -0.07572, -0.17572, 0.877861, 2.00204, 2.38204, 3.18204, 2.82204, 3.20204, 2.95459, 2.07791, 1.81791, 2.34711, 4.33711, 5.13507, 5.6936, 5.6936, 5.0386, 3.3986, 0.0886, -2.07502, -2.28502, -3.84042],
        )
        self.declare_parameter(
            "waypoints_y",
            [-1.79182, -2.97463, -3.97463, -4.20463, -3.96463, -2.88463, -1.76463, -0.45463, 0.83537, 1.88882, 3.01298, 3.16298, 2.67802, 0.58802, -1.90198, -3.12453, -4.00121, -4.47121, -5.18839, -5.18839, -4.99043, -3.2919, 3.8681, 5.00256, 5.16256, 5.16256, 2.99909, -0.36091, -1.81742],
        )
        self.declare_parameter("waypoints_lb", [])
        self.declare_parameter("waypoints_ub", [])

        self.control_period = float(self.get_parameter("control_period").value)
        self.horizon = int(self.get_parameter("horizon").value)
        self.ref_speed = float(self.get_parameter("reference_speed").value)
        self.ay_max = float(self.get_parameter("ay_max").value)
        self.max_steer_deg = abs(float(self.get_parameter("max_steering_angle_deg").value))
        q_ey = float(self.get_parameter("q_ey").value)
        q_epsi = float(self.get_parameter("q_epsi").value)
        q_t = float(self.get_parameter("q_t").value)
        qn_ey = float(self.get_parameter("qn_ey").value)
        qn_epsi = float(self.get_parameter("qn_epsi").value)
        qn_t = float(self.get_parameter("qn_t").value)
        r_speed = float(self.get_parameter("r_speed").value)
        r_steer = float(self.get_parameter("r_steer").value)
        self.osqp_max_iter = int(self.get_parameter("osqp_max_iter").value)
        self.osqp_eps_abs = float(self.get_parameter("osqp_eps_abs").value)
        self.osqp_eps_rel = float(self.get_parameter("osqp_eps_rel").value)
        self.wp_ordered_mode = bool(self.get_parameter("wp_ordered_mode").value)
        self.wp_ordered_start_index = int(self.get_parameter("wp_ordered_start_index").value)
        self.wp_reached_distance = max(0.05, float(self.get_parameter("wp_reached_distance").value))
        self.wp_pass_margin = max(0.0, float(self.get_parameter("wp_pass_margin").value))
        self.debug_decisions = bool(self.get_parameter("debug_decisions").value)
        self.debug_decisions_every = int(self.get_parameter("debug_decisions_every").value)
        if self.debug_decisions_every <= 0:
            self.debug_decisions_every = 1

        self._tick_counter = 0
        self._ordered_wp_id: int | None = None

        wp_x = [float(v) for v in self.get_parameter("waypoints_x").value]
        wp_y = [float(v) for v in self.get_parameter("waypoints_y").value]
        wp_lb_raw = [float(v) for v in self.get_parameter("waypoints_lb").value]
        wp_ub_raw = [float(v) for v in self.get_parameter("waypoints_ub").value]

        wp_lb: list[float] | None = None
        wp_ub: list[float] | None = None
        if wp_lb_raw or wp_ub_raw:
            if len(wp_lb_raw) != len(wp_x) or len(wp_ub_raw) != len(wp_x):
                raise ValueError(
                    "waypoints_lb and waypoints_ub must be either empty or have the same "
                    "length as waypoints_x/waypoints_y"
                )
            wp_lb = wp_lb_raw
            wp_ub = wp_ub_raw
        else:
            self.get_logger().warn(
                "No waypoints_lb/waypoints_ub provided; using adaptive lateral bounds "
                "derived from waypoint spacing."
            )

        self.reference_path = RosReferencePath(
            wp_x=wp_x,
            wp_y=wp_y,
            v_ref=self.ref_speed,
            wp_lb=wp_lb,
            wp_ub=wp_ub,
            circular=True,
        )

        self.model = BicycleModel(
            reference_path=self.reference_path,
            length=0.257, #wheelbase of the TT02 in meters
            width=0.188, #width of the TT02 in meters
            Ts=self.control_period,
        )
        self.model.external_waypoint_sync = True

        self.max_steer_rad = math.radians(self.max_steer_deg)
        self.max_speed = float(GilbertDriverGeneric.SPEED_LIMIT_FORWARD)
        self.min_speed = float(GilbertDriverGeneric.SPEED_LIMIT_REVERSE)

        q = sparse.diags([q_ey, q_epsi, q_t])
        r = sparse.diags([r_speed, r_steer])
        qn = sparse.diags([qn_ey, qn_epsi, qn_t])

        solver_settings = {
            "max_iter": self.osqp_max_iter,
            "eps_abs": self.osqp_eps_abs,
            "eps_rel": self.osqp_eps_rel,
            "verbose": False,
            "warm_start": True,
            "polish": False,
            "adaptive_rho": True,
        }

        input_constraints = {
            "umin": np.array([self.min_speed, -math.tan(self.max_steer_rad) / self.model.length]),
            "umax": np.array([self.max_speed, math.tan(self.max_steer_rad) / self.model.length]),
        }
        state_constraints = {
            "xmin": np.array([-np.inf, -np.inf, -np.inf]),
            "xmax": np.array([np.inf, np.inf, np.inf]),
        }

        self.mpc = MPC(
            model=self.model,
            N=self.horizon,
            Q=q,
            R=r,
            QN=qn,
            StateConstraints=state_constraints,
            InputConstraints=input_constraints,
            ay_max=self.ay_max,
            SolverSettings=solver_settings,
        )

        self.latest_pose: tuple[float, float, float] | None = None

        self.create_subscription(Odometry, self.TOPIC_ODOM, self._on_odometry, 10)
        self.command_publisher = self.create_publisher(AckermannDrive, self.TOPIC_COMMAND, 10)
        self.create_timer(self.control_period, self._on_control_tick)

        self.get_logger().info(
            "MPC node started: publishing Ackermann on /car/command from /odom feedback"
        )
        self.get_logger().info(
            f"MPC limits: ay_max={self.ay_max:.3f} m/s^2, max_steer={self.max_steer_deg:.2f} deg"
        )
        self.get_logger().info(
            f"MPC debug_decisions={self.debug_decisions} every={self.debug_decisions_every} ticks"
        )

    def _on_odometry(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        self.latest_pose = (x, y, yaw)

    def _waypoint_distance(self, wp_id: int, x: float, y: float) -> float:
        wp = self.reference_path.waypoints[wp_id]
        return float(math.hypot(wp.x - x, wp.y - y))

    def _find_nearest_waypoint_index(self, x: float, y: float) -> tuple[int, float]:
        n_wp = len(self.reference_path.waypoints)
        distances = [self._waypoint_distance(i, x, y) for i in range(n_wp)]
        nearest_wp_id = int(np.argmin(distances))
        return nearest_wp_id, float(distances[nearest_wp_id])

    def _sync_model_with_odometry(self) -> tuple[int, float]:
        assert self.latest_pose is not None
        x, y, yaw = self.latest_pose

        if self.wp_ordered_mode:
            n_wp = len(self.reference_path.waypoints)
            nearest_wp_id, _ = self._find_nearest_waypoint_index(x, y)

            if self._ordered_wp_id is None:
                if 0 <= self.wp_ordered_start_index < n_wp:
                    self._ordered_wp_id = self.wp_ordered_start_index
                else:
                    self._ordered_wp_id = nearest_wp_id

            selected_wp_id = self._ordered_wp_id
            selected_wp_dist = self._waypoint_distance(selected_wp_id, x, y)

            # Strict ordered traversal with robust progression:
            # - advance when entering waypoint acceptance radius
            # - also advance if next waypoint is clearly closer (current waypoint
            #   likely missed/passed), which avoids orbiting around one target.
            reached_radius = self.wp_reached_distance + self.wp_pass_margin
            for _ in range(n_wp):
                next_wp_id = (selected_wp_id + 1) % n_wp
                next_wp_dist = self._waypoint_distance(next_wp_id, x, y)

                reached_by_radius = selected_wp_dist <= reached_radius
                passed_current_wp = next_wp_dist + self.wp_pass_margin < selected_wp_dist

                if not (reached_by_radius or passed_current_wp):
                    break

                selected_wp_id = next_wp_id
                selected_wp_dist = next_wp_dist

            self._ordered_wp_id = selected_wp_id
        else:
            selected_wp_id, selected_wp_dist = self._find_nearest_waypoint_index(x, y)

        self.model.wp_id = selected_wp_id
        self.model.current_waypoint = self.reference_path.waypoints[selected_wp_id]
        self.model.s = float(self.reference_path.cumulative_lengths[selected_wp_id])

        self.model.temporal_state = TemporalState(x=x, y=y, psi=yaw)
        return selected_wp_id, selected_wp_dist

    def _on_control_tick(self) -> None:
        if self.latest_pose is None:
            return

        self._tick_counter += 1
        selected_wp_id, selected_wp_dist = self._sync_model_with_odometry()

        try:
            u = self.mpc.get_control()
        except Exception as exc:
            self.get_logger().warn(f"MPC solve failed: {exc}")
            if self.debug_decisions:
                self.get_logger().warn(traceback.format_exc())
            # Do not keep stale/high command when solver fails.
            stop_msg = AckermannDrive()
            stop_msg.speed = 0.0
            stop_msg.steering_angle = 0.0
            self.command_publisher.publish(stop_msg)
            return

        speed_cmd = float(np.clip(u[0], self.min_speed, self.max_speed))
        steer_cmd = float(np.clip(u[1], -self.max_steer_rad, self.max_steer_rad))

        msg = AckermannDrive()
        msg.speed = speed_cmd
        msg.steering_angle = steer_cmd
        self.command_publisher.publish(msg)

        if self.debug_decisions and self._tick_counter % self.debug_decisions_every == 0:
            x, y, yaw = self.latest_pose
            active_wp_id = int(self.model.wp_id)
            active_wp = self.reference_path.waypoints[active_wp_id]

            e_y = float(self.model.spatial_state.e_y)
            e_psi = float(self.model.spatial_state.e_psi)

            self.get_logger().info(
                "MPC decision | "
                f"pose=({x:.2f},{y:.2f},{yaw:.2f}) "
                f"mode={'ordered' if self.wp_ordered_mode else 'nearest'} "
                f"target_wp={selected_wp_id} (wp#{selected_wp_id + 1}) d={selected_wp_dist:.2f} "
                f"active_wp={active_wp_id} wp=({active_wp.x:.2f},{active_wp.y:.2f}) "
                f"wp_psi={active_wp.psi:.2f} wp_kappa={active_wp.kappa:.3f} wp_vref={active_wp.v_ref:.2f} "
                f"errors=(e_y={e_y:.2f},e_psi={e_psi:.2f}) "
                f"u_raw=(v={u[0]:.2f},delta={u[1]:.2f}) u_cmd=(v={speed_cmd:.2f},delta={steer_cmd:.2f})"
            )

    def destroy_node(self) -> bool:
        stop_msg = AckermannDrive()
        stop_msg.speed = 0.0
        stop_msg.steering_angle = 0.0
        try:
            if rclpy.ok():
                self.command_publisher.publish(stop_msg)
        except Exception:
            pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TT02MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
