"""
Camera and LiDAR fusion module for Super Mega Fusion.
"""

import numpy as np
import networkx as nx
from networkx import DiGraph
from typing import TypeAlias, cast, Optional
from dataclasses import dataclass
from scipy.ndimage import binary_dilation, distance_transform_edt
from enum import Enum
from common import SegmentationLabels
from matplotlib import pyplot as plt

# ====================================================
#  Types and Constants
# ====================================================

# ===== Base =====
OccupancyGrid: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
OccupancyGridLabelled: TypeAlias = np.ndarray[
    tuple[int, int], np.dtype[np.uint8]
]  # uint8: SegmentationLabels values
FlowField: TypeAlias = tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],  # forward_col
    np.ndarray[tuple[int, int], np.dtype[np.float64]],  # forward_row
]
Graph: TypeAlias = "DiGraph[tuple[int, int]]"


@dataclass
class Pose2D:
    x: float  # m
    y: float  # m
    theta: float  # rad

    @staticmethod
    def change_frame(
        local_pose: "Pose2D",
        frame_origin_pose: "Pose2D",
    ) -> "Pose2D":
        """Convert a pose from a local frame to the global frame defined by frame_origin.

        Args:
            local_pose: The pose in the local frame
            frame_origin: The origin of the local frame in global coordinates
        Returns:
            The pose in the global frame
        """
        # Formula : |x_new|   |a|              |x_old|                       |a|
        #           |y_new| = |b| + R(t_old) *  |y_old| , where local_pose = |b|
        #           |t_new|   t_old + r                                      |r|

        # rotation matrix from local to global
        c, s = np.cos(frame_origin_pose.theta), np.sin(frame_origin_pose.theta)
        xn = frame_origin_pose.x + c * local_pose.x - s * local_pose.y
        yn = frame_origin_pose.y + s * local_pose.x + c * local_pose.y
        tn = (frame_origin_pose.theta + local_pose.theta) % (2 * np.pi)
        return Pose2D(x=xn, y=yn, theta=tn)

    def to_global(self, frame_origin_pose: "Pose2D") -> "Pose2D":
        """Get coordinates of this pose in the global frame defined by frame_origin.

        Args:
            frame_origin_pose: The origin of the local frame in global coordinates
        Returns:
            The pose in the global frame
        """
        return Pose2D.change_frame(self, frame_origin_pose)


class Command:

    def __init__(
        self,
        steering_angle: float,
        speed: float,
        sample_frequency_hz: float,
        sub_steps: int,
        wheelbase_m: float,
    ) -> None:
        self.steering_angle = steering_angle
        self.speed = speed
        positions = self._get_trajectory_positions(
            sample_frequency=sample_frequency_hz,
            sub_steps=sub_steps,
            wheelbase_m=wheelbase_m,
        )
        self.positions: list[Pose2D] = positions

    def _get_trajectory_positions(
        self, sample_frequency: float, sub_steps: int, wheelbase_m: float
    ) -> list[Pose2D]:
        """Compute collision-check samples for this command."""
        # guard against invalid parameters
        if sample_frequency <= 0 or sub_steps <= 0:
            return []

        # at least one sample (the endpoint)
        sub_steps = max(sub_steps, 1)

        # determine turning radius from steering angle; the formula matches the
        # one used in ``path_finding_pipeline_flat``.  A zero steering angle
        # is treated as straight motion (infinite radius).
        if self.steering_angle == 0.0:
            radius = float("inf")
        else:
            radius = wheelbase_m / np.tan(self.steering_angle)

        speed = self.speed

        period = 1.0 / sample_frequency  # total time for one planning step
        dt = period / sub_steps  # time between samples
        poses: list[Pose2D] = []
        for i in range(1, sub_steps + 1):  # skip t=0 (current position)
            t = i * dt
            dist = speed * t
            if radius == float("inf"):
                # straight ahead along +X
                x = dist
                y = 0.0
                theta = 0.0
            else:
                ang = dist / abs(radius)
                # left turn (radius>0) gives positive theta, right turn
                # (radius<0) gives negative.
                x = abs(radius) * np.sin(ang)
                y = radius * (1 - np.cos(ang))
                theta = np.sign(radius) * ang
            poses.append(Pose2D(x=x, y=y, theta=theta))
        return poses


@dataclass
class TrajectoryPoint:
    pose: Pose2D  # Pose at this point
    command: (
        Command | None
    )  # Command chosen to reach this point from the previous one (None for the first point)
    remaining_commands: list[int]  # indices in PLANNING_COMMANDS of the commands


class TrackWallDirection(Enum):
    RED_RIGHT_GREEN_LEFT = 0
    RED_LEFT_GREEN_RIGHT = 1


# ====================================================
#  Helpers
# ====================================================


def pose_to_pixel(
    pose: Pose2D, pixels_per_meter: float, w: int, h: int
) -> tuple[int, int]:
    """Convert a world-frame pose to grid pixel coords (col, row).

    This version rounds to integers and is intended for indexing the occupancy
    grid.  Other plotting functions should use :func:`pose_to_plot` to avoid
    quantisation artefacts."""
    px = int(np.round(pose.x * pixels_per_meter + w / 2))
    py = int(np.round(-pose.y * pixels_per_meter + h / 2))
    return (px, py)


def pose_to_pixel_float(
    pose: Pose2D, pixels_per_meter: float, w: int, h: int
) -> tuple[float, float]:
    """Convert a world-frame pose to grid pixel coords (col, row).

    This version returns floating-point coordinates and is intended for plotting
    purposes."""
    px = pose.x * pixels_per_meter + w / 2
    py = -pose.y * pixels_per_meter + h / 2
    return (px, py)

# ====================================================
#  Implementation
# ====================================================


def occupancy_grid_preprocess(
    occupancy_grid: OccupancyGridLabelled,
    dilation_radius: int = 2,
) -> OccupancyGridLabelled:
    """Preprocess the occupancy grid for path planning."""
    binary_grid = occupancy_grid != SegmentationLabels.FREE.value
    dilated_binary = cast(
        OccupancyGrid, binary_dilation(binary_grid, iterations=dilation_radius)
    )
    introduced_obstacles = dilated_binary & ~binary_grid
    # Add introduced obstacles to the original grid, marking them as MISC_OBSTACLE
    dilated_grid = occupancy_grid.copy()
    dilated_grid[introduced_obstacles] = SegmentationLabels.MISC_OBSTACLE
    return dilated_grid


def make_graph_and_flow_from_occupancy_grid(
    occupancy_grid: OccupancyGridLabelled,
    track_wall_direction: TrackWallDirection,
    enable_diagonal: bool = True,
) -> tuple["Graph", FlowField]:
    """Make a graph from the occupancy grid for path planning."""

    # ===== Compute flow field =====
    # Get the distance to the nearest red/green wall
    distance_to_red_wall = cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        distance_transform_edt(occupancy_grid != SegmentationLabels.WALL_RED.value),
    )
    distance_to_green_wall = cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        distance_transform_edt(occupancy_grid != SegmentationLabels.WALL_GREEN.value),
    )

    # Flow field
    f_cross = (
        distance_to_red_wall - distance_to_green_wall
        if track_wall_direction == TrackWallDirection.RED_RIGHT_GREEN_LEFT
        else distance_to_green_wall - distance_to_red_wall
    )
    # np.gradient returns [∂f/∂row, ∂f/∂col]
    grad_row, grad_col = cast(
        list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
        np.gradient(f_cross),
    )
    # Forward direction
    forward_col = -grad_row
    forward_row = grad_col

    # ===== Build graph =====
    circuit_graph: Graph = DiGraph()
    h, w = occupancy_grid.shape
    cardinal = ((1, 0), (-1, 0), (0, 1), (0, -1))
    diagonal = ((1, 1), (1, -1), (-1, 1), (-1, -1))

    FREE = SegmentationLabels.FREE.value
    for y in range(h):
        for x in range(w):
            if occupancy_grid[y, x] != FREE:
                continue
            circuit_graph.add_node((x, y))

            # check if edge direction is consistent with forward flow
            def _forward_ok(ddx: int, ddy: int) -> bool:
                return ddx * forward_col[y, x] + ddy * forward_row[y, x] >= -1e-3

            # add cardinal edges (unit cost)
            for dx, dy in cardinal:
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < w and 0 <= ny_ < h and occupancy_grid[ny_, nx_] == FREE:
                    if _forward_ok(dx, dy):
                        circuit_graph.add_edge((x, y), (nx_, ny_), weight=1.0)
            # add diagonal edges (√2 cost)
            if enable_diagonal:
                for dx, dy in diagonal:
                    nx_, ny_ = x + dx, y + dy
                    if (
                        0 <= nx_ < w
                        and 0 <= ny_ < h
                        and occupancy_grid[ny_, nx_] == FREE
                    ):
                        # check if both adjacent cardinal cells are free to allow diagonal movement
                        if (
                            occupancy_grid[y + dy, x] == FREE
                            and occupancy_grid[y, x + dx] == FREE
                        ):
                            if _forward_ok(dx, dy):
                                circuit_graph.add_edge(
                                    (x, y), (nx_, ny_), weight=np.sqrt(2)
                                )
    return circuit_graph, (forward_col, forward_row)


def naive_path_planning(
    graph: Graph, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    """Naive path planning algorithm using NetworkX shortest path.

    Args:
        graph: The graph to search, with nodes as (x, y) pixel coordinates.
        start: The starting pixel coordinate (x, y).
        goal: The target pixel coordinate (x, y).
    Returns:
        List of (x, y) pixel coordinates along the shortest path from start to goal
        according to the weights stored in the graph edges. If no path exists
        or either endpoint is missing, an empty list is returned."""
    try:
        # NetworkX will automatically use the "weight" attribute if present.
        return nx.shortest_path(graph, source=start, target=goal, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def naive_position_prediction(
    naive_path: list[tuple[int, int]],
    opponent_speed_pps: float,  # pixels per second
    pixels_per_meter: float,
    sample_frequency_hz: float = 1.0,  # how many samples per second
    grid_shape: Optional[tuple[int, int]] = None,
) -> list[Pose2D]:
    """Naive position prediction along the naive path.

    Args:
        naive_path: List of pixel coordinates along the path (col, row).
        opponent_speed_pps: Speed of the opponent in pixels per second.
        pixels_per_meter: Conversion factor from pixels to meters.
        sample_frequency_hz: How many samples to generate per second.
        grid_shape: Optional shape of the occupancy grid (height, width) for recentering.
    Returns:
        List of estimated poses along the path, sampled at regular time intervals.
    """
    if not naive_path or opponent_speed_pps <= 0.0:
        return []

    # start from raw pixel coordinates
    pts_px = np.array(naive_path, dtype=float)
    # recenter around grid middle if shape is known
    if grid_shape is not None:
        h, w = grid_shape
        # integer division keeps the origin consistent with pose_to_grid_cell
        # (which also truncates via int()) and with draw_figure (which uses //)
        centre = np.array([w // 2, h // 2], dtype=float)
        pts_px -= centre

    # flip y: pixel y increases downward, but world y grows upward
    pts_px[:, 1] = -pts_px[:, 1]
    # convert to metres
    pts = pts_px / pixels_per_meter
    # convert speed (pixels per second) to metres per second
    opponent_speed_mps = opponent_speed_pps / pixels_per_meter
    # time between samples (s)
    dt = 1.0 / sample_frequency_hz if sample_frequency_hz > 0 else 1.0
    # distance advanced between samples (m)
    step_dist = opponent_speed_mps * dt

    if pts.shape[0] == 1:
        # single point only
        return [Pose2D(x=pts[0, 0], y=pts[0, 1], theta=0.0)]

    seg_vecs = pts[1:] - pts[:-1]  # vectors between consecutive points (m)
    seg_lengths = np.linalg.norm(seg_vecs, axis=1)
    total_length = seg_lengths.sum()
    if total_length == 0.0:
        return [Pose2D(x=pts[0, 0], y=pts[0, 1], theta=0.0)]

    # compute cumulative length at each vertex
    cum = np.concatenate(([0.0], np.cumsum(seg_lengths)))  # length N

    # sampling distances along path (include end explicitly)
    samples = np.arange(0.0, total_length, step_dist)
    if samples.size == 0 or samples[-1] != total_length:
        samples = np.append(samples, total_length)

    # determine segment index for each sample
    # seg_idx[i] gives index of segment containing samples[i]
    seg_idx = np.searchsorted(cum, samples, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_lengths) - 1)

    # compute local distance within segment
    local_dist = samples - cum[seg_idx]
    seg_len = seg_lengths[seg_idx]
    t = np.zeros_like(local_dist)
    nonzero = seg_len > 0
    t[nonzero] = local_dist[nonzero] / seg_len[nonzero]

    # compute positions and orientations vectorized
    starts = pts[seg_idx]
    vecs = seg_vecs[seg_idx]
    xs = starts[:, 0] + t * vecs[:, 0]
    ys = starts[:, 1] + t * vecs[:, 1]
    thetas = np.arctan2(vecs[:, 1], vecs[:, 0]).astype(float)

    estimated_positions = [
        Pose2D(x=float(x), y=float(y), theta=float(th))
        for x, y, th in zip(xs, ys, thetas)
    ]

    return estimated_positions


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------


def trajectory_to_positions(
    trajectory: list[TrajectoryPoint],
) -> list[Pose2D]:
    """Convert a trajectory of command endpoints into a dense pose list.

    Args:
        trajectory: Output of :func:`super_mega_fusion_path_planning`.
    Returns:
        A list of ``Pose2D`` instances sampled along the executed commands in
        chronological order (including the initial pose)."""
    if not trajectory:
        return []

    poses: list[Pose2D] = [trajectory[0].pose]
    current = trajectory[0].pose
    for tp in trajectory[1:]:
        if tp.command is None:
            continue
        for local in tp.command.positions:
            global_p = local.to_global(current)
            poses.append(global_p)
        current = poses[-1]
    return poses

def super_mega_fusion_path_planning(
    occupancy_grid: OccupancyGridLabelled,
    position_start: Pose2D,
    planning_commands: list[Command],
    opponent_collision_radius_m: float,
    trajectory_plan_length: int,
    pixels_per_meter: float,
    estimated_positions_seti: Optional[list[Pose2D]] = None,
    estimated_positions_opponents: Optional[list[list[Pose2D]]] = None,
    max_iterations: int = 200_000,
    backtrack_count: int = 1,
    closest_path_bias_strength: float = 1.0,
    command_priority_bias_strength: float = 1.0,
) -> list[TrajectoryPoint]:
    """Super Mega Fusion path planning algorithm.
    Args:
        occupancy_grid: The preprocessed occupancy grid (2D array of SegmentationLabels).
        position_start: The starting pose of the vehicle in world coordinates.
        estimated_positions_seti: Optional list of estimated future poses for SETI, used for A* biasing.
        estimated_positions_opponents: Optional list of lists of estimated future poses for opponents, used for collision checking.
        trajectory_plan_length: The number of trajectory points to plan (including the start point).

    Returns:
        The best (longest collision-free) trajectory found.
    """
    h, w = occupancy_grid.shape
    n_commands = len(planning_commands)

    def _get_all_command_indices() -> list[int]:
        return list(range(n_commands))

    def _clone_trajectory(
        traj: list[TrajectoryPoint],
    ) -> list[TrajectoryPoint]:
        return [
            TrajectoryPoint(
                pose=Pose2D(p.pose.x, p.pose.y, p.pose.theta),
                command=p.command,
                remaining_commands=_get_all_command_indices(),
            )
            for p in traj
        ]

    # Initialise trajectory with the start position
    trajectory: list[TrajectoryPoint] = [
        TrajectoryPoint(
            pose=position_start,
            command=None,
            remaining_commands=_get_all_command_indices(),
        )
    ]
    best_trajectory = _clone_trajectory(trajectory)

    seti_ref = estimated_positions_seti or []
    opponents = estimated_positions_opponents or []

    iter_counter = 0
    print(f"Starting path planning procedure for {trajectory_plan_length} points...")
    while len(trajectory) < trajectory_plan_length and iter_counter < max_iterations:
        iter_counter += 1
        if (iter_counter % 100_000) == 0:
            cur = trajectory[-1]
            print(
                f"Iteration {iter_counter}, "
                f"tl={len(trajectory)}, "
                f"trl={len(cur.remaining_commands)}, "
                f"c={cur.command}"
            )

        # ---- pick / backtrack ----
        current_point = trajectory[-1]
        if not current_point.remaining_commands:
            if current_point.command is None:
                # Initial position exhausted all commands
                break
            else:
                k_back = min(backtrack_count, len(trajectory) - 1)
                for _ in range(k_back):
                    trajectory.pop()  # backtrack
                continue

        # ---- biased command selection ----
        n_remaining = len(current_point.remaining_commands)
        step_idx = len(trajectory)  # time step we are planning for

        if n_remaining > 1:
            scores = np.zeros(n_remaining)

            # -- A* reference bias: prefer commands whose endpoint is closer
            #    to the reference trajectory at this time step --
            if seti_ref and step_idx < len(seti_ref) and closest_path_bias_strength > 0:
                ref_pose = seti_ref[step_idx]
                astar_scores = np.empty(n_remaining)
                for ci, cmd_idx in enumerate(current_point.remaining_commands):
                    local_last = planning_commands[cmd_idx].positions[-1]
                    ep = local_last.to_global(current_point.pose)
                    astar_scores[ci] = -np.hypot(ep.x - ref_pose.x, ep.y - ref_pose.y)
                score_range = astar_scores.max() - astar_scores.min()
                if score_range > 0:
                    astar_scores = (astar_scores - astar_scores.min()) / score_range
                scores += astar_scores * closest_path_bias_strength

            # -- Priority bias: prefer lower-index commands (faster / more
            #    direct options are listed first in planning_commands) --
            if command_priority_bias_strength > 0:
                # priority_score goes from 1.0 (index 0) down to 0.0 (last index),
                # normalised over the full command list so the scale is consistent.
                denom = max(n_commands - 1, 1)
                priority_scores = np.array(
                    [
                        1.0 - current_point.remaining_commands[ci] / denom
                        for ci in range(n_remaining)
                    ]
                )
                scores += priority_scores * command_priority_bias_strength

            # deterministic selection: choose the command with highest weighted score
            # softmax is not required; compare raw scores directly for argmax.
            # if multiple commands tie, the first (lowest index) is picked automatically by argmax.
            next_command_index = int(np.argmax(scores))
        else:
            next_command_index = 0

        next_cmd_idx = current_point.remaining_commands.pop(next_command_index)
        next_command = planning_commands[next_cmd_idx]

        # ---- simulate trajectory for this command ----
        global_samples = [
            p.to_global(current_point.pose) for p in next_command.positions
        ]

        # ---- collision check: occupancy grid ----
        collision = False
        for p in global_samples:
            px, py = pose_to_pixel(p, pixels_per_meter, w, h)
            if px < 0 or px >= w or py < 0 or py >= h:
                collision = True
                break
            elif occupancy_grid[py, px] != SegmentationLabels.FREE.value:
                collision = True
                break
        if collision:
            continue

        # ---- collision check: opponents ----
        for opp_poses in opponents:
            if len(opp_poses) > len(trajectory):
                opp_pose = opp_poses[len(trajectory)]
                for p in global_samples:
                    dist = np.hypot(p.x - opp_pose.x, p.y - opp_pose.y)
                    if dist < opponent_collision_radius_m:
                        collision = True
                        break
            if collision:
                break
        if collision:
            continue

        # ---- append new point (endpoint only) ----
        trajectory.append(
            TrajectoryPoint(
                pose=next_command.positions[-1].to_global(current_point.pose),
                command=next_command,
                remaining_commands=_get_all_command_indices(),
            )
        )
        if len(trajectory) > len(best_trajectory):
            best_trajectory = _clone_trajectory(trajectory)

    print(f"Planned trajectory with {len(best_trajectory)} points!")
    return best_trajectory


def draw_figure(
    occupancy_grid: OccupancyGridLabelled,
    pixels_per_meter: float,
    collision_radius_m: float,
    positions_seti: Optional[list[Pose2D]] = None,
    positions_naive_seti: Optional[list[Pose2D]] = None,
    positions_opponents: Optional[list[list[Pose2D]]] = None,
    start_position_seti: Optional[Pose2D] = None,
    start_positions_opponents: Optional[list[Pose2D]] = None,
    target_pose: Optional[Pose2D] = None,
    misc_scatter_points: Optional[list[tuple[int, int]]] = None,
    flow_field: Optional[FlowField] = None,
    do_draw_origin: bool = True,
    scale_icons: float = 1.0,
    title: str = "Full picture",
) -> None:
    """Draw a figure showing the occupancy grid and trajectories."""

    # Pad figure to be squared
    h, w = occupancy_grid.shape
    size = max(h, w)
    pad_h, pad_w = size - h, size - w
    if pad_h or pad_w:
        pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        occupancy_grid = np.pad(
            occupancy_grid, pad, constant_values=SegmentationLabels.FREE.value
        )
        h, w = occupancy_grid.shape
        if flow_field is not None:
            f_col, f_row = flow_field
            f_col = np.pad(f_col, pad, constant_values=0.0)
            f_row = np.pad(f_row, pad, constant_values=0.0)
            flow_field = (f_col, f_row)

    image_display = np.zeros((h, w, 3), dtype=np.uint8)
    image_display[:, :] = [200, 200, 200]  # Default to gray
    image_display[occupancy_grid == SegmentationLabels.FREE.value] = [
        255,
        255,
        255,
    ]
    image_display[occupancy_grid == SegmentationLabels.WALL_RED.value] = [
        255,
        0,
        0,
    ]
    image_display[occupancy_grid == SegmentationLabels.WALL_GREEN.value] = [
        0,
        255,
        0,
    ]
    # draw grid + origin
    plt.imshow(image_display)
    # keep axes square so one pixel in x equals one pixel in y
    plt.gca().set_aspect("equal", adjustable="box")
    origin_x = w // 2
    origin_y = h // 2
    if do_draw_origin:
        plt.scatter(origin_x, origin_y, c="cyan", marker="x", s=50, label="Origin")

    # Draw flow field as quiver plot
    if flow_field is not None:
        forward_col, forward_row = flow_field
        skip = max(1, min(h, w) // 30)  # adjust density of quiver arrows
        Y, X = np.mgrid[0:h:skip, 0:w:skip]
        U = forward_col[::skip, ::skip]*scale_icons
        V = forward_row[::skip, ::skip]*scale_icons
        plt.quiver(
            X,
            Y,
            U,
            V,
            color="magenta",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003*scale_icons/2,
            alpha=0.3,
            label="Flow Field",
        )

    # Misc data
    if misc_scatter_points:
        for i, cell in enumerate(misc_scatter_points):
            plt.scatter(cell[0], cell[1], c="orange", label=f"Point {i+1}")

    # Precompute viridis colors for SETI/opponent poses
    viridis_colors: Optional[np.ndarray[tuple[int, 4], np.dtype[np.float64]]] = None
    if positions_seti or positions_opponents or positions_naive_seti:
        step_count = max(
            len(positions_seti) if positions_seti else 0,
            (
                max((len(poses) for poses in positions_opponents), default=0)
                if positions_opponents
                else 0
            ),
            len(positions_naive_seti) if positions_naive_seti else 0,
        )
        cmap = plt.get_cmap("viridis")
        viridis_colors = cmap(np.linspace(0, 1, step_count))

    def plot_arrow(pose: Pose2D, color: str, label: Optional[str] = None) -> None:
        x_px, y_px = pose_to_pixel_float(pose, pixels_per_meter, w, h)
        dx = np.cos(pose.theta) * 1.0 * scale_icons  # arrow length in pixels
        dy = -np.sin(pose.theta) * 1.0 * scale_icons
        plt.arrow(
            x_px,
            y_px,
            dx,
            dy,
            head_width=1.0 * scale_icons,
            head_length=1.2 * scale_icons,
            fc=color,
            ec=color,
            alpha=0.7,
            length_includes_head=True,
        )
        if label:
            plt.plot([], [], c=color, label=label)

    # Plot SETI trajectory
    if positions_seti:
        # Plot dotted path (float coords to avoid snapping)
        xs: list[float] = []
        ys: list[float] = []
        for pose in positions_seti:
            x_px, y_px = pose_to_pixel_float(pose, pixels_per_meter, w, h)
            xs.append(x_px)
            ys.append(y_px)
        plt.plot(xs, ys, c="blue", linestyle="--", alpha=0.7, label="SETI Path")
        # Plot arrows for each pose using viridis_colors when available.
        # skip some samples to avoid clutter
        skip = max(1, len(positions_seti) // 30)
        for idx in range(0, len(positions_seti), skip):
            pose = positions_seti[idx]
            color = "blue"
            if viridis_colors is not None and idx < len(viridis_colors):
                c = viridis_colors[idx]
                color = (float(c[0]), float(c[1]), float(c[2]))
            plot_arrow(pose, color=color)

    # plot SETI naive trajectory as a dotted line with semi‑transparent alpha
    if positions_naive_seti:
        xs = []
        ys = []
        for pose in positions_naive_seti:
            x_px, y_px = pose_to_pixel_float(pose, pixels_per_meter, w, h)
            xs.append(x_px)
            ys.append(y_px)
        # use a single label for the entire path
        plt.plot(xs, ys, c="blue", linestyle=":", alpha=0.5, label="Naive SETI Path")

    # plot each opponent trajectory similarly
    if positions_opponents:
        # Draw trajectories
        for j, poses in enumerate(positions_opponents):
            xs = []
            ys = []
            for pose in poses:
                x_px, y_px = pose_to_pixel_float(pose, pixels_per_meter, w, h)
                xs.append(x_px)
                ys.append(y_px)
            plt.plot(
                xs,
                ys,
                c="red",
                linestyle=":",
                alpha=0.5,
                label="Opponent paths" if j == 0 else None,  # label only the first for legend clarity
            )

            # Also scatter disks of collision_radius_m, with color viridis based on
            if viridis_colors is not None and poses:
                for i, pose in enumerate(poses):
                    x_px, y_px = pose_to_pixel_float(pose, pixels_per_meter, w, h)
                    circle = plt.Circle(
                        (x_px, y_px),
                        collision_radius_m * pixels_per_meter,
                        color=viridis_colors[i],
                        alpha=0.3,
                        label="Opponent collision" if (j, i) == (0, 0) else None,
                    )
                    plt.gca().add_patch(circle)

    # Plot start positions
    if start_position_seti:
        plot_arrow(start_position_seti, "blue", label="SETI Start")
    if start_positions_opponents:
        for j, start_pos in enumerate(start_positions_opponents):
            plot_arrow(start_pos, "orange", label="Opponents start" if j == 0 else None)
    # Plot target if given (plus marker rather than arrow)
    if target_pose:
        tx, ty = pose_to_pixel_float(target_pose, pixels_per_meter, w, h)
        plt.scatter(tx, ty, c="green", marker="+", s=100, label="Target")

    plt.legend(loc="lower left")
    plt.title(title)
    plt.show()


# ====================================================
#  Tests
# ====================================================
if __name__ == "__main__":
    from pathlib import Path
    from matplotlib import pyplot as plt
    from PIL import Image

    # ====== Parameters ======

    WHEELBASE_M = 0.5  # m distance between front and rear axles
    PIXELS_PER_METER = 15  # pixels for one meter
    COLLISION_RADIUS_M = (
        1 / PIXELS_PER_METER
    )  # radius of collision area around the obstacles
    OPPONENT_COLLISION_RADIUS_M = 0.1  # m, collision radius for opponent avoidance
    OPPONENT_SPEED_MPS = 0.5  # meters per second, for naive position prediction
    SAMPLE_FREQUENCY_HZ = 4.0  # how often to sample along the path (Hz)
    COMMAND_TRAJECTORY_SUBSTEPS = (
        5  # how many trajectory points to precompute for each command
    )
    BACKTRACK_COUNT = 1  # how many steps to backtrack during path planning

    # Planning
    sfh, cts, wb = SAMPLE_FREQUENCY_HZ, COMMAND_TRAJECTORY_SUBSTEPS, WHEELBASE_M
    PLANNING_COMMANDS: list[Command] = [  #
        Command(0.0,             1.0, sfh, cts, wb),  # straight
        Command(np.radians(15),  0.8, sfh, cts, wb),  # slight left
        Command(np.radians(-15), 0.8, sfh, cts, wb),  # slight right
        Command(0.0,             0.3, sfh, cts, wb),  # straight slow
        Command(np.radians(30),  0.5, sfh, cts, wb),  # left
        Command(np.radians(-30), 0.5, sfh, cts, wb),  # right
        Command(np.radians(60),  0.3, sfh, cts, wb),  # sharp left
        Command(np.radians(-60), 0.3, sfh, cts, wb),  # sharp right
    ]
    MAX_ITERATIONS = 200_000  # maximum iterations for the path planning search
    TRAJECTORY_PLAN_LENGTH = 70  # default number of trajectory points to plan

    CLOSEST_PATH_BIAS_STRENGTH = 10  # softmax temperature scale for A* reference bias
    COMMAND_PRIORITY_BIAS_STRENGTH = (
        0  # softmax scale favouring lower-index (faster) commands
    )

    # Start positions
    position_start_seti = Pose2D(x=-1.0, y=-0.2, theta=np.pi / 6)
    position_start_oponents = [
        Pose2D(x=-0.7, y=0.1, theta=0),
        Pose2D(x=0.1, y=0.3, theta=np.pi / 2),
    ]
    position_target = Pose2D(x=0.8, y=-2, theta=0.0)

    # Load occupancy grid from image
    print("Loading occupancy grid...")
    img = Image.open(Path(__file__).parent / "test_resources" / "carte.png").convert(
        "RGB"
    )
    img_array = np.array(img, dtype=np.uint8)  # shape (H, W, 3)

    # Convert to occupancy grid with labels
    print("Converting to occupancy grid with labels...")
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    occupancy_grid_labelled = cast(
        OccupancyGridLabelled,
        np.full(r.shape, SegmentationLabels.MISC_OBSTACLE.value, dtype=np.uint8),
    )
    occupancy_grid_labelled[(r == 255) & (g == 255) & (b == 255)] = (
        SegmentationLabels.FREE.value
    )
    occupancy_grid_labelled[(r == 255) & (g == 0) & (b == 0)] = (
        SegmentationLabels.WALL_RED.value
    )
    occupancy_grid_labelled[(r == 0) & (g == 255) & (b == 0)] = (
        SegmentationLabels.WALL_GREEN.value
    )

    if False:
        print("Unique values in occupancy grid:", np.unique(occupancy_grid_labelled))

    # Test occupancy_grid_preprocess
    print("Testing occupancy_grid_preprocess...")
    occupancy_grid_labelled = occupancy_grid_preprocess(
        occupancy_grid_labelled,
        dilation_radius=int(np.ceil(COLLISION_RADIUS_M * PIXELS_PER_METER)),
    )
    if False:
        plt.imshow(occupancy_grid_labelled, cmap="gray")
        plt.title("Preprocessed Occupancy Grid")
        plt.show()

    # Test make_graph_from_occupancy_grid
    print("Testing make_graph_from_occupancy_grid...")
    graph, flow_field = make_graph_and_flow_from_occupancy_grid(
        occupancy_grid_labelled, TrackWallDirection.RED_RIGHT_GREEN_LEFT
    )
    if False:
        print(
            f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
        )

    H, W = occupancy_grid_labelled.shape
    cell_start_seti = pose_to_pixel(position_start_seti, PIXELS_PER_METER, W, H)
    cell_start_opponents = [
        pose_to_pixel(pos, PIXELS_PER_METER, W, H) for pos in position_start_oponents
    ]
    cell_target = pose_to_pixel(position_target, PIXELS_PER_METER, W, H)
    cell_origin = pose_to_pixel(Pose2D(x=0, y=0, theta=0), PIXELS_PER_METER, W, H)

    # Test naive_path_planning
    print("Testing naive_path_planning...")
    path_seti = naive_path_planning(graph, cell_start_seti, cell_target)
    path_opponents = [
        naive_path_planning(graph, cell_start, cell_target)
        for cell_start in cell_start_opponents
    ]
    # Test naive_position_prediction
    print("Testing naive_position_prediction...")
    speed_pps = OPPONENT_SPEED_MPS * PIXELS_PER_METER  # pixels per second
    positions_seti = naive_position_prediction(
        path_seti,
        speed_pps,
        PIXELS_PER_METER,
        sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
        grid_shape=occupancy_grid_labelled.shape,
    )
    positions_opponents = [
        naive_position_prediction(
            path,
            speed_pps,
            pixels_per_meter=PIXELS_PER_METER,
            sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
            grid_shape=occupancy_grid_labelled.shape,
        )
        for path in path_opponents
    ]

    # Test super_mega_fusion_path_planning
    print("Testing super_mega_fusion_path_planning...")
    seti_trajectory = super_mega_fusion_path_planning(
        occupancy_grid_labelled,
        position_start_seti,
        PLANNING_COMMANDS,
        OPPONENT_COLLISION_RADIUS_M,
        TRAJECTORY_PLAN_LENGTH,
        PIXELS_PER_METER,
        estimated_positions_seti=positions_seti,
        estimated_positions_opponents=positions_opponents,
        max_iterations=MAX_ITERATIONS,
        backtrack_count=BACKTRACK_COUNT,
        closest_path_bias_strength=CLOSEST_PATH_BIAS_STRENGTH,
        command_priority_bias_strength=COMMAND_PRIORITY_BIAS_STRENGTH,
    )

    # Display results
    if True:
        # expand the coarse endpoint trajectory into a dense set of samples so
        # the drawn path appears smooth rather than zig‑zaggy
        smooth_seti = trajectory_to_positions(seti_trajectory)

        draw_figure(
            occupancy_grid_labelled,
            PIXELS_PER_METER,
            COLLISION_RADIUS_M,
            positions_naive_seti=positions_seti,
            positions_seti=smooth_seti,
            positions_opponents=positions_opponents,
            start_position_seti=position_start_seti,
            start_positions_opponents=position_start_oponents,
            target_pose=position_target,
            flow_field=flow_field,
            # misc_scatter_points=[],
            title="Path planner",
        )
