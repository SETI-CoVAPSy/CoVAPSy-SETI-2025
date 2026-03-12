from __future__ import annotations

import argparse
import math
import signal
import subprocess
import shutil
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from PIL import Image, ImageDraw
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from scipy.ndimage import convolve, label


class LapMonitor(Node):
    def __init__(self) -> None:
        super().__init__("lap_monitor")
        self.subscription = self.create_subscription(Odometry, "/odom", self._odom_callback, 10)
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self._scan_callback, 10)

        self.has_pose = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0

        self.start_set = False
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_theta = 0.0

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.path_len_m = 0.0
        self.motion_logged = False

        self.has_scan = False
        self.last_ranges: list[float] = []
        self.scan_angle_min = -math.pi
        self.scan_angle_increment = 2.0 * math.pi / 360.0

        self.start_line_ready = False
        self.start_line_width_m = 0.0
        self.start_line_half_width_m = 0.0
        self.prev_line_side: float | None = None

    def _scan_callback(self, msg: LaserScan) -> None:
        self.last_ranges = list(msg.ranges)
        self.scan_angle_min = float(msg.angle_min)
        self.scan_angle_increment = float(msg.angle_increment)
        self.has_scan = True

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

    def _odom_callback(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self.current_x = float(pose.position.x)
        self.current_y = float(pose.position.y)
        self.current_theta = self._yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        if not self.start_set:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.start_theta = self.current_theta
            self.prev_x = self.current_x
            self.prev_y = self.current_y
            self.start_set = True
            self.get_logger().info(
                f"Départ: x={self.start_x:.3f}, y={self.start_y:.3f}, theta={self.start_theta:.3f}"
            )
        else:
            step = math.hypot(self.current_x - self.prev_x, self.current_y - self.prev_y)
            self.path_len_m += step
            self.prev_x = self.current_x
            self.prev_y = self.current_y
            if (not self.motion_logged) and self.path_len_m >= 0.20:
                self.motion_logged = True
                self.get_logger().info(
                    f"Mouvement détecté: path={self.path_len_m:.2f}m, x={self.current_x:.3f}, y={self.current_y:.3f}"
                )

        self.has_pose = True

    def lap_metrics(self) -> tuple[float, float]:
        if not self.start_set:
            return float("inf"), float("inf")
        distance_to_start = math.hypot(self.current_x - self.start_x, self.current_y - self.start_y)
        heading_delta = self._angle_diff(self.current_theta, self.start_theta)
        return distance_to_start, heading_delta

    def _sector_min(self, min_deg: float, max_deg: float) -> float:
        if not self.has_scan or not self.last_ranges:
            return float("inf")

        vals: list[float] = []
        for i, r in enumerate(self.last_ranges):
            if not math.isfinite(r) or r <= 0.02:
                continue
            a = self.scan_angle_min + i * self.scan_angle_increment
            a_deg = a * 180.0 / math.pi
            if min_deg <= a_deg <= max_deg:
                vals.append(float(r))

        if not vals:
            return float("inf")
        vals.sort()
        idx = max(0, min(len(vals) - 1, int(0.25 * (len(vals) - 1))))
        return vals[idx]

    def try_init_start_line(self) -> bool:
        if self.start_line_ready or not self.start_set or not self.has_scan:
            return self.start_line_ready

        left_d = self._sector_min(65.0, 115.0)
        right_d = self._sector_min(-115.0, -65.0)
        if not (math.isfinite(left_d) and math.isfinite(right_d)):
            return False

        width = left_d + right_d
        if width < 0.30 or width > 5.0:
            return False

        self.start_line_width_m = width
        self.start_line_half_width_m = 0.5 * width
        self.start_line_ready = True
        self.get_logger().info(
            f"Ligne départ initialisée: largeur~{self.start_line_width_m:.2f}m"
        )
        return True

    def start_line_state(self) -> tuple[float, float]:
        """Returns (signed distance to line normal, lateral offset along line tangent)."""
        dx = self.current_x - self.start_x
        dy = self.current_y - self.start_y

        nx = math.cos(self.start_theta)
        ny = math.sin(self.start_theta)
        tx = -math.sin(self.start_theta)
        ty = math.cos(self.start_theta)

        side = dx * nx + dy * ny
        lateral = dx * tx + dy * ty
        return side, lateral


def run_cmd(cmd: list[str]) -> bool:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return False
    return result.returncode == 0


def log_info(node: Node, message: str) -> None:
    if rclpy.ok():
        node.get_logger().info(message)
    else:
        print(f"[INFO] [lap_monitor]: {message}")


def log_warn(node: Node, message: str) -> None:
    if rclpy.ok():
        node.get_logger().warn(message)
    else:
        print(f"[WARN] [lap_monitor]: {message}")


def log_error(node: Node, message: str) -> None:
    if rclpy.ok():
        node.get_logger().error(message)
    else:
        print(f"[ERROR] [lap_monitor]: {message}")


def save_map_with_retry(base_path: Path, attempts: int = 3, wait_s: float = 1.0) -> bool:
    for i in range(attempts):
        ok = run_cmd([
            "ros2", "run", "nav2_map_server", "map_saver_cli", "-f", str(base_path)
        ])
        if ok:
            return True
        if i < attempts - 1:
            time.sleep(wait_s)
    return False


def stop_slam() -> bool:
    if run_cmd(["ros2", "lifecycle", "set", "/slam_toolbox", "shutdown"]):
        return True
    return run_cmd(["pkill", "-f", "slam_toolbox"])


def _sample_evenly(items: list[tuple[float, Path]], max_items: int) -> list[tuple[float, Path]]:
    if max_items <= 0:
        return []
    if len(items) <= max_items:
        return items

    n = len(items)
    idxs = [round(i * (n - 1) / (max_items - 1)) for i in range(max_items)]
    uniq: list[int] = []
    for idx in idxs:
        if not uniq or uniq[-1] != idx:
            uniq.append(idx)
    if uniq[-1] != n - 1:
        uniq[-1] = n - 1
    return [items[i] for i in uniq]


def evaluate_track_closure(
    carte_path: Path,
    *,
    min_wall_pixels: int,
    min_largest_ratio: float,
    max_dangling_ratio: float,
) -> tuple[bool, float, str]:
    try:
        img = Image.open(carte_path).convert("RGB")
    except Exception:
        return False, 0.0, "lecture image impossible"

    data = np.asarray(img, dtype=np.uint8)
    red = np.all(data == np.array([255, 0, 0], dtype=np.uint8), axis=2)
    green = np.all(data == np.array([0, 255, 0], dtype=np.uint8), axis=2)

    neigh_kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    def wall_stats(mask: np.ndarray) -> tuple[bool, float, int, float, float]:
        pix = int(np.count_nonzero(mask))
        if pix == 0:
            return False, 0.0, 0, 0.0, 1.0

        labels, n_comp = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        if n_comp <= 0:
            return False, 0.0, 0, 0.0, 1.0

        counts = np.bincount(labels.ravel())
        largest = int(counts[1:].max()) if counts.size > 1 else 0
        largest_ratio = float(largest) / float(max(1, pix))

        neigh = convolve(mask.astype(np.uint8), neigh_kernel, mode="constant", cval=0)
        dangling = int(np.count_nonzero(mask & (neigh <= 1)))
        dangling_ratio = float(dangling) / float(max(1, pix))

        area_score = min(1.0, float(pix) / float(max(1, min_wall_pixels)))
        largest_score = min(1.0, largest_ratio / max(1e-6, min_largest_ratio))
        dangling_score = min(1.0, max_dangling_ratio / max(1e-6, dangling_ratio))
        score = 0.25 * area_score + 0.45 * largest_score + 0.30 * dangling_score

        good = (
            pix >= min_wall_pixels
            and largest_ratio >= min_largest_ratio
            and dangling_ratio <= max_dangling_ratio
        )
        return good, score, pix, largest_ratio, dangling_ratio

    red_ok, red_score, red_pix, red_largest, red_dangling = wall_stats(red)
    green_ok, green_score, green_pix, green_largest, green_dangling = wall_stats(green)

    closed = red_ok and green_ok
    score = 0.5 * (red_score + green_score)
    detail = (
        f"R(pix={red_pix},largest={red_largest:.2f},dang={red_dangling:.3f}) "
        f"G(pix={green_pix},largest={green_largest:.2f},dang={green_dangling:.3f})"
    )
    return closed, score, detail


def build_timeline_figure(
    snapshots: list[tuple[float, Path]],
    output_path: Path,
    *,
    max_frames: int,
    cols: int,
    stop_reason: str,
) -> bool:
    valid = [(t, p) for t, p in snapshots if p.exists()]
    if not valid:
        return False

    selected = _sample_evenly(valid, max(1, max_frames))
    images: list[Image.Image] = []
    labels: list[str] = []
    for idx, (t, p) in enumerate(selected):
        images.append(Image.open(p).convert("RGB"))
        tag = f"t={t:.1f}s"
        if idx == len(selected) - 1:
            tag += " (final)"
        labels.append(tag)

    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    panel_w = max(img.width for img in images)
    panel_h = max(img.height for img in images)

    pad = 16
    title_h = 40
    caption_h = 24
    width = pad + cols * (panel_w + pad)
    height = title_h + pad + rows * (panel_h + caption_h + pad)

    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (pad, 10),
        f"SLAM map evolution ({len(selected)} frames, stop: {stop_reason})",
        fill=(0, 0, 0),
    )

    for i, (img, label_txt) in enumerate(zip(images, labels)):
        r = i // cols
        c = i % cols
        x0 = pad + c * (panel_w + pad)
        y0 = title_h + pad + r * (panel_h + caption_h + pad)
        x_img = x0 + (panel_w - img.width) // 2
        y_img = y0 + (panel_h - img.height) // 2
        canvas.paste(img, (x_img, y_img))
        draw.text((x0, y0 + panel_h + 4), label_txt, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Live SLAM map export with automatic stop on lap completion")
    parser.add_argument("--period", type=float, default=5.0, help="Live save period in seconds")
    parser.add_argument("--line-cross-tol", type=float, default=0.08, help="Signed-distance tolerance to validate line crossing (m)")
    parser.add_argument("--line-cross-hysteresis", type=float, default=0.18, help="Minimum |side| on both sides of the line to validate a real crossing (m)")
    parser.add_argument("--line-width-margin", type=float, default=0.25, help="Extra margin around estimated start-line width (m)")
    parser.add_argument("--line-away-tol", type=float, default=0.25, help="Distance to line needed to arm return detection (m)")
    parser.add_argument("--start-return-radius", type=float, default=0.80, help="Distance to start position to validate lap completion (m), tolerant to odom drift")
    parser.add_argument("--start-exit-radius", type=float, default=0.90, help="Distance to start position required before lap completion can trigger (m)")
    parser.add_argument("--min-lap-path", type=float, default=3.0, help="Minimum traveled distance (m) before lap completion can trigger")
    parser.add_argument(
        "--map-closure-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable smart stop when track map appears closed and clean",
    )
    parser.add_argument(
        "--require-lap-before-map-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a full-lap geometric detection before allowing map-closure stop",
    )
    parser.add_argument(
        "--map-closure-consecutive",
        type=int,
        default=3,
        help="Number of consecutive closed-map detections required before stop",
    )
    parser.add_argument(
        "--map-closure-min-wall-pixels",
        type=int,
        default=280,
        help="Minimum pixels per wall contour to consider map closure",
    )
    parser.add_argument(
        "--map-closure-min-largest-ratio",
        type=float,
        default=0.88,
        help="Largest connected-component ratio threshold per wall contour",
    )
    parser.add_argument(
        "--map-closure-max-dangling-ratio",
        type=float,
        default=0.03,
        help="Maximum dangling-endpoint ratio per wall contour",
    )
    parser.add_argument(
        "--map-style",
        choices=["trinary", "track-walls"],
        default="track-walls",
        help="Output style passed to create_carte_from_slam.py (default: track-walls)",
    )
    parser.add_argument(
        "--min-wall-area",
        type=int,
        default=120,
        help="Minimum wall component area for track-walls style",
    )
    parser.add_argument(
        "--wall-thickness-px",
        type=int,
        default=2,
        help="Wall contour thickness in pixels for track-walls style",
    )
    parser.add_argument(
        "--timeline-out",
        type=Path,
        default=Path("/workspaces/CoVAPSy-SETI-2025/Software/super_mega_fusion/figure_evolution_slam.png"),
        help="Output image for timeline grid of SLAM map evolution",
    )
    parser.add_argument(
        "--timeline-max-frames",
        type=int,
        default=12,
        help="Maximum number of timeline frames displayed in the output image",
    )
    parser.add_argument(
        "--timeline-cols",
        type=int,
        default=4,
        help="Number of columns in timeline output image",
    )
    args = parser.parse_args()

    root = Path("/workspaces/CoVAPSy-SETI-2025")
    pf = root / "Software" / "path_finding"
    tmp_base = pf / "slam_live_tmp"
    final_base = pf / "slam_map"
    live_png = pf / "carte_live.png"
    final_png = pf / "carte.png"
    final_fig = root / "Software" / "super_mega_fusion" / "figure_finale.png"
    converter = pf / "create_carte_from_slam.py"

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = LapMonitor()
    stop_requested = False

    def _handle_sigint(signum, frame):  # type: ignore[no-untyped-def]
        nonlocal stop_requested
        stop_requested = True

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    start_time = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    live_history_dir = pf / "carte_live_history" / run_id
    live_history_dir.mkdir(parents=True, exist_ok=True)
    snapshots: list[tuple[float, Path]] = []
    lap_armed = False
    exited_start_zone = False
    lap_gate_reached = False
    map_closed_streak = 0
    last_save_t = 0.0
    stop_reason = "auto-stop"

    while rclpy.ok():
        if stop_requested:
            stop_reason = "arrêt manuel (Ctrl+C)"
            log_info(node, "Ctrl+C reçu: sauvegarde finale...")
            break

        rclpy.spin_once(node, timeout_sec=0.1)

        now = time.time()
        elapsed = now - start_time

        if node.start_set:
            node.try_init_start_line()
            should_stop = False
            dist_start, _ = node.lap_metrics()
            if node.start_line_ready:
                side, lateral = node.start_line_state()
                width_limit = node.start_line_half_width_m + args.line_width_margin

                if not lap_armed and abs(side) >= max(args.line_away_tol, 2.0 * args.line_cross_tol):
                    lap_armed = True
                    node.prev_line_side = side
                    log_info(node, f"[LINE] Détection armée: side={side:.3f}m, lateral={lateral:.3f}m")

            if lap_armed and (not exited_start_zone) and dist_start >= args.start_exit_radius:
                exited_start_zone = True
                log_info(node, f"[LINE] Zone départ quittée: dist_start={dist_start:.3f}m")

            if lap_armed and exited_start_zone and node.start_line_ready and node.path_len_m >= args.min_lap_path:
                side, lateral = node.start_line_state()
                prev_side = node.prev_line_side
                node.prev_line_side = side

                crossed_line = (
                    prev_side is not None
                    and (prev_side * side) < 0.0
                    and abs(prev_side) >= args.line_cross_hysteresis
                    and abs(side) >= args.line_cross_hysteresis
                    and abs(lateral) <= width_limit
                )
                if crossed_line:
                    should_stop = True
                    stop_reason = (
                        f"2e franchissement ligne départ (lat={lateral:.2f}m, side={side:.2f}m, width={node.start_line_width_m:.2f}m)"
                    )

            if should_stop:
                if args.map_closure_stop and args.require_lap_before_map_stop:
                    if not lap_gate_reached:
                        lap_gate_reached = True
                        log_info(
                            node,
                            (
                                f"Tour complet détecté: elapsed={elapsed:.1f}s, path={node.path_len_m:.2f}m, "
                                f"raison={stop_reason}. En attente fermeture/propreté carte..."
                            ),
                        )
                else:
                    log_info(
                        node,
                        f"Tour complet détecté: elapsed={elapsed:.1f}s, path={node.path_len_m:.2f}m, raison={stop_reason}"
                    )
                    break

        if now - last_save_t >= args.period:
            ok_save = run_cmd([
                "ros2", "run", "nav2_map_server", "map_saver_cli", "-f", str(tmp_base)
            ])
            if ok_save:
                ok_convert = run_cmd([
                    "python3", str(converter),
                    "--pgm", str(tmp_base) + ".pgm",
                    "--out", str(live_png),
                    "--save-figure", str(live_png),
                    "--style", args.map_style,
                    "--min-wall-area", str(max(1, args.min_wall_area)),
                    "--wall-thickness-px", str(max(1, args.wall_thickness_px)),
                ])
                if ok_convert and live_png.exists():
                    snap = live_history_dir / f"carte_t_{int(max(0.0, elapsed) * 10):06d}.png"
                    shutil.copy2(live_png, snap)
                    snapshots.append((elapsed, snap))
                    log_info(node, f"[LIVE] Updated {live_png.name}")

                    if args.map_closure_stop and args.map_style == "track-walls":
                        map_closed, map_score, map_detail = evaluate_track_closure(
                            live_png,
                            min_wall_pixels=max(1, args.map_closure_min_wall_pixels),
                            min_largest_ratio=max(0.1, min(0.999, args.map_closure_min_largest_ratio)),
                            max_dangling_ratio=max(1e-4, min(0.5, args.map_closure_max_dangling_ratio)),
                        )
                        if map_closed:
                            map_closed_streak += 1
                        else:
                            map_closed_streak = 0

                        lap_gate_ok = (not args.require_lap_before_map_stop) or lap_gate_reached
                        if map_closed and map_closed_streak >= max(1, args.map_closure_consecutive) and node.path_len_m >= args.min_lap_path and lap_gate_ok:
                            stop_reason = (
                                f"circuit fermé/propre détecté (score={map_score:.2f}, streak={map_closed_streak})"
                            )
                            log_info(node, f"[MAP] {stop_reason} | {map_detail}")
                            break
                else:
                    log_warn(node, "[LIVE] conversion to carte_live.png failed")
            else:
                log_warn(node, "[LIVE] map_saver_cli failed (SLAM not ready yet?)")
            last_save_t = now

    try:
        run_cmd([
            "ros2", "topic", "pub", "--once", "/car/command",
            "ackermann_msgs/msg/AckermannDrive",
            "{speed: 0.0, steering_angle: 0.0}"
        ])
    except Exception:
        pass

    try:
        elapsed_total = time.time() - start_time
        ok_final = save_map_with_retry(final_base, attempts=3, wait_s=1.0)
        if ok_final:
            ok_final_convert = run_cmd([
                "python3", str(converter),
                "--pgm", str(final_base) + ".pgm",
                "--out", str(final_png),
                "--save-figure", str(final_fig),
                "--style", args.map_style,
                "--min-wall-area", str(max(1, args.min_wall_area)),
                "--wall-thickness-px", str(max(1, args.wall_thickness_px)),
            ])
            if ok_final_convert and final_png.exists():
                final_snap = live_history_dir / "carte_finale.png"
                shutil.copy2(final_png, final_snap)
                snapshots.append((elapsed_total, final_snap))
                timeline_ok = build_timeline_figure(
                    snapshots,
                    args.timeline_out,
                    max_frames=max(2, args.timeline_max_frames),
                    cols=max(1, args.timeline_cols),
                    stop_reason=stop_reason,
                )
                log_info(node, f"[END] {stop_reason}")
                log_info(node, f"[OK] Final map: {final_png}")
                log_info(node, f"[OK] Final figure: {final_fig}")
                if timeline_ok:
                    log_info(node, f"[OK] Timeline figure: {args.timeline_out}")
                else:
                    log_warn(node, "[END] Timeline figure not generated (no valid snapshots)")
            else:
                log_error(node, "Final map conversion failed")
        else:
            log_error(node, "Final map save failed (vérifie que slam_toolbox tourne encore)")

        if stop_slam():
            log_info(node, "[OK] SLAM arrêté")
        else:
            log_warn(node, "[WARN] Impossible d'arrêter SLAM automatiquement")

    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
