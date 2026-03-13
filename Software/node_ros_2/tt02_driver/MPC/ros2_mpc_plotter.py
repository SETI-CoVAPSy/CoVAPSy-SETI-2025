"""Realtime matplotlib plotter for MPC diagnostics."""

from __future__ import annotations

import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan

from .tools import laser_scan_to_world_frame, read_lidar_calibration


class MPCRealtimePlotter(Node):
    """Plot waypoints, walls from lidar, car pose and MPC commands in realtime."""

    TOPIC_ODOM = "/odom"
    TOPIC_SCAN = "/scan"
    TOPIC_COMMAND = "/car/command"

    def __init__(self) -> None:
        super().__init__("tt02_mpc_plotter")

        self.declare_parameter("plot_rate_hz", 10.0)
        self.declare_parameter("history_size", 1200)
        self.declare_parameter("command_history_seconds", 20.0)
        self.declare_parameter("scan_stride", 3)
        dyn_num = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("scan_angle_offset_deg", 0.0, dyn_num)
        self.declare_parameter("scan_mirror", True)
        self.declare_parameter("scan_reverse", False)
        self.declare_parameter("lidar_offset_x", 0.0, dyn_num)
        self.declare_parameter("lidar_offset_y", 0.0, dyn_num)
        self.declare_parameter("axis_padding", 1.0)
        self.declare_parameter("waypoints_x", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("waypoints_y", Parameter.Type.DOUBLE_ARRAY)

        self.plot_rate_hz = max(1.0, float(self.get_parameter("plot_rate_hz").value))
        history_size = max(50, int(self.get_parameter("history_size").value))
        self.command_history_seconds = max(
            2.0, float(self.get_parameter("command_history_seconds").value)
        )
        self.scan_stride = max(1, int(self.get_parameter("scan_stride").value))
        self.axis_padding = max(0.1, float(self.get_parameter("axis_padding").value))
        initial_calibration = read_lidar_calibration(self)

        wp_x = [float(v) for v in self.get_parameter("waypoints_x").value]
        wp_y = [float(v) for v in self.get_parameter("waypoints_y").value]
        if len(wp_x) != len(wp_y):
            raise ValueError("waypoints_x and waypoints_y must have same length")

        self.waypoints = np.column_stack((wp_x, wp_y)) if wp_x else np.empty((0, 2), dtype=float)

        self.pose: tuple[float, float, float] | None = None
        self.last_cmd_speed = 0.0
        self.last_cmd_steer = 0.0

        self.history_x: deque[float] = deque(maxlen=history_size)
        self.history_y: deque[float] = deque(maxlen=history_size)
        self.scan_world_points = np.empty((0, 2), dtype=float)

        self.cmd_t: deque[float] = deque()
        self.cmd_v: deque[float] = deque()
        self.cmd_delta: deque[float] = deque()

        self.create_subscription(Odometry, self.TOPIC_ODOM, self._on_odom, 10)
        self.create_subscription(LaserScan, self.TOPIC_SCAN, self._on_scan, 10)
        self.create_subscription(AckermannDrive, self.TOPIC_COMMAND, self._on_command, 10)

        self._init_figure()
        self.create_timer(1.0 / self.plot_rate_hz, self._refresh_plot)

        self.get_logger().info(
            f"MPC plotter started: rate={self.plot_rate_hz:.1f}Hz scan_stride={self.scan_stride}"
        )
        self.get_logger().info(
            "Lidar calibration: "
            f"offset={math.degrees(initial_calibration.angle_offset_rad):.1f}deg "
            f"mirror={initial_calibration.mirror} reverse={initial_calibration.reverse} "
            f"sensor_offset=({initial_calibration.offset_x:.2f},{initial_calibration.offset_y:.2f})m"
        )

    def _init_figure(self) -> None:
        plt.ion()
        self.fig, (self.ax_map, self.ax_cmd) = plt.subplots(
            2,
            1,
            figsize=(11, 8),
            gridspec_kw={"height_ratios": [3.0, 1.0]},
        )
        self.fig.canvas.manager.set_window_title("TT02 MPC Realtime Plot")

        self.ax_map.set_title("Map: waypoints, walls (lidar), and car pose")
        self.ax_map.set_xlabel("X [m]")
        self.ax_map.set_ylabel("Y [m]")
        self.ax_map.grid(True, alpha=0.25)
        self.ax_map.axis("equal")

        if self.waypoints.size > 0:
            self.ax_map.plot(
                self.waypoints[:, 0],
                self.waypoints[:, 1],
                "o-",
                color="#2E86DE",
                markersize=3,
                linewidth=1.2,
                label="Waypoints",
            )

        self.wall_scatter = self.ax_map.scatter(
            [], [], s=6, c="#7F8C8D", alpha=0.55, label="Walls from lidar"
        )
        (self.history_line,) = self.ax_map.plot(
            [], [], color="#27AE60", linewidth=1.6, label="Car history"
        )
        (self.car_marker,) = self.ax_map.plot([], [], "o", color="#C0392B", markersize=7, label="Car")
        self.heading = self.ax_map.quiver(
            [0.0], [0.0], [1.0], [0.0],
            angles="xy", scale_units="xy", scale=1.0, color="#C0392B", width=0.005
        )
        self.info_text = self.ax_map.text(
            0.02,
            0.98,
            "",
            transform=self.ax_map.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        self.ax_map.legend(loc="upper right")

        self.ax_cmd.set_title("MPC commands")
        self.ax_cmd.set_xlabel("Time [s]")
        self.ax_cmd.set_ylabel("Command")
        self.ax_cmd.grid(True, alpha=0.25)
        (self.speed_line,) = self.ax_cmd.plot([], [], color="#8E44AD", linewidth=1.6, label="Speed [m/s]")
        (self.steer_line,) = self.ax_cmd.plot([], [], color="#D35400", linewidth=1.6, label="Steer [rad]")
        self.ax_cmd.legend(loc="upper right")

        if self.waypoints.size > 0:
            x_min = float(np.min(self.waypoints[:, 0])) - self.axis_padding
            x_max = float(np.max(self.waypoints[:, 0])) + self.axis_padding
            y_min = float(np.min(self.waypoints[:, 1])) - self.axis_padding
            y_max = float(np.max(self.waypoints[:, 1])) + self.axis_padding
            self.ax_map.set_xlim(x_min, x_max)
            self.ax_map.set_ylim(y_min, y_max)

        self.fig.tight_layout()
        plt.show(block=False)

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        self.pose = (x, y, yaw)
        self.history_x.append(x)
        self.history_y.append(y)

    def _on_scan(self, msg: LaserScan) -> None:
        if self.pose is None or not msg.ranges:
            return

        calibration = read_lidar_calibration(self)
        self.scan_world_points = laser_scan_to_world_frame(
            msg,
            self.pose,
            calibration,
            stride=self.scan_stride,
        )

    def _on_command(self, msg: AckermannDrive) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        self.last_cmd_speed = float(msg.speed)
        self.last_cmd_steer = float(msg.steering_angle)

        self.cmd_t.append(now)
        self.cmd_v.append(self.last_cmd_speed)
        self.cmd_delta.append(self.last_cmd_steer)

        threshold = now - self.command_history_seconds
        while self.cmd_t and self.cmd_t[0] < threshold:
            self.cmd_t.popleft()
            self.cmd_v.popleft()
            self.cmd_delta.popleft()

    def _refresh_plot(self) -> None:
        if not plt.fignum_exists(self.fig.number):
            return

        if self.scan_world_points.size > 0:
            self.wall_scatter.set_offsets(self.scan_world_points)
        else:
            self.wall_scatter.set_offsets(np.empty((0, 2), dtype=float))

        if self.history_x:
            hx = np.asarray(self.history_x, dtype=float)
            hy = np.asarray(self.history_y, dtype=float)
            self.history_line.set_data(hx, hy)

        if self.pose is not None:
            x, y, yaw = self.pose
            self.car_marker.set_data([x], [y])
            self.heading.set_offsets(np.array([[x, y]], dtype=float))
            self.heading.set_UVC(np.array([math.cos(yaw)]), np.array([math.sin(yaw)]))

            self.info_text.set_text(
                f"Car: x={x:.2f} y={y:.2f} yaw={yaw:.2f} rad\n"
                f"MPC cmd: v={self.last_cmd_speed:.2f} m/s  delta={self.last_cmd_steer:.2f} rad\n"
                f"Lidar points: {int(self.scan_world_points.shape[0])}"
            )

        if self.cmd_t:
            t = np.asarray(self.cmd_t, dtype=float)
            t_rel = t - t[-1]
            v = np.asarray(self.cmd_v, dtype=float)
            d = np.asarray(self.cmd_delta, dtype=float)
            self.speed_line.set_data(t_rel, v)
            self.steer_line.set_data(t_rel, d)
            self.ax_cmd.set_xlim(-self.command_history_seconds, 0.0)

            y_all = np.concatenate((v, d))
            y_min = float(np.min(y_all)) - 0.1
            y_max = float(np.max(y_all)) + 0.1
            if abs(y_max - y_min) < 1e-4:
                y_min -= 0.5
                y_max += 0.5
            self.ax_cmd.set_ylim(y_min, y_max)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MPCRealtimePlotter()
    try:
        while rclpy.ok() and plt.fignum_exists(node.fig.number):
            rclpy.spin_once(node, timeout_sec=0.05)
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        plt.close("all")
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
