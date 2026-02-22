"""
Path finding pipeline bien moche
"""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.axes import Axes
from typing import Callable, Union, TypedDict, TypeAlias
import networkx as nx
from networkx import Graph
from scipy.ndimage import binary_dilation
from dataclasses import dataclass
from enum import Enum

# =======================================
#  Parameters
# =======================================

FACTOR_PIXEL_PER_METER = 15  # 15 pixels per meter, so 1 pixel = 0.0667 m
BINARY_DILATION_PIXELS = (
    1  # number of pixels to dilate obstacles in occupancy grid (inflation radius)
)
ASTAR_DO_DIAGONAL = (
    True  # whether to allow diagonal moves in A* graph (8-connected vs 4-connected)
)
ASTAR_AVERAGE_SPEED_MPS = (
    1.0  # assumed average speed of opponents for time estimation (m/s)
)
SAMPLE_INTERVAL_S = 0.3   # s, how often to sample points along A* path / random tree for position estimation

WHEELBASE_M = 0.5  # m, distance between front and rear axles for trajectory prediction
# old distance-based sampling constant removed; time interval SAMPLE_INTERVAL_S now used
PATH_PLANNING_HORIZON_S = 50.0  # s, how far into the future to plan paths

# Angle convention: **counter-clockwise positive** (standard math)
#   theta = 0   → facing +X (right)
#   theta = π/2 → facing +Y (up)
#   Positive steering angle → left turn (CCW)


def get_positions() -> "InitialPositions":
    # For now, just return some hardcoded positions
    return {
        "seti": Pose2D(x=-1.0, y=-0.2, theta=np.pi / 6),
        "opponents": [
            Pose2D(x=-0.7, y=0.1, theta=0),
            Pose2D(x=0.1, y=0.3, theta=np.pi / 2),
        ],
        "target": Pose2D(x=1.0, y=2.0, theta=0.0),
    }


# =======================================
#  Definitions
# =======================================

# plt figure generator that will generate a figure given figures appended to show
# An item is either an image array, a callable that draws onto an Axes, or a matplotlib Figure
FigureItem = Union[NDArray, Callable[[Axes], None], Figure]


def figure_to_array(fig: Figure) -> NDArray[np.uint8]:
    """Render a matplotlib Figure to an RGBA NumPy array using the Agg backend (no GUI)."""
    FigureCanvas(fig)  # attach Agg canvas if not already present
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return np.asarray(buf)


# plt figure generator that will generate a figure given figures appended to show
class FigureGenerator:
    def __init__(self):
        self.figures: list[tuple[str, FigureItem]] = []

    def add_figure(self, name: str, figure: FigureItem) -> None:
        """Add an image (NDArray) or a draw function (Callable[[Axes], None])."""
        self.figures.append((name, figure))

    def show(self):
        # Create figure with subplots for each figure in self.figures
        n = len(self.figures)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        main_fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten() if n > 1 else [axes]
        for ax, (name, item) in zip(axes, self.figures):
            ax.set_title(name)
            if callable(item):
                item(ax)  # Let the callable draw directly onto the axis
            elif isinstance(item, Figure):
                ax.imshow(figure_to_array(item))
                ax.axis("off")
            else:
                ax.imshow(item)
                ax.axis("off")
        plt.tight_layout()
        plt.show()


def get_occupancy_grid():
    # load carte.png
    img = Image.open(Path(__file__).parent / "carte.png")
    img_array = np.array(img, dtype=np.bool_)  # And invert
    img_array = np.rot90(img_array, 2)
    img_array = np.flip(img_array, 1)
    return img_array


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
        """Convert this pose from a local frame to the global frame defined by frame_origin.
        
        Args:
            frame_origin_pose: The origin of the local frame in global coordinates
        Returns:
            The pose in the global frame
        """
        return Pose2D.change_frame(self, frame_origin_pose)



class InitialPositions(TypedDict):
    seti: Pose2D
    opponents: list[Pose2D]
    target: Pose2D


# =======================================
#  Main
# =======================================
fig_gen = FigureGenerator()

# ======= Initial positions and map =======
# Get occupancy grid
occupancy_grid = get_occupancy_grid()

# compute extent in pixel coordinates so that origin (0,0) → center of image
h, w = occupancy_grid.shape
extent = (-w / 2, w / 2, -h / 2, h / 2)

fig_occ = Figure()
ax_occ = fig_occ.add_subplot(111)
# origin='lower' makes y increase upward which matches usual coordinate frames
ax_occ.imshow(
    np.bitwise_not(occupancy_grid), cmap="gray", extent=extent, origin="lower"
)
ax_occ.set_xlim(extent[0], extent[1])
ax_occ.set_ylim(extent[2], extent[3])
fig_gen.add_figure("Occupancy Grid", fig_occ)

# Apply grow operation to inflate obstacles by 1 pixel (0.0667 m) to account for vehicle size and safety margin
occupancy_grid = binary_dilation(occupancy_grid, iterations=BINARY_DILATION_PIXELS)

# Get positions
positions = get_positions()


# Plot positions on occupancy grid
# Draw each vehicle as an arrow oriented by its theta field
def draw_positions(ax: Axes):
    # redraw occupancy grid with same extent so axes are centered
    h, w = occupancy_grid.shape
    extent = (-w / 2, w / 2, -h / 2, h / 2)
    ax.imshow(
        np.bitwise_not(occupancy_grid), cmap="gray", extent=extent, origin="lower"
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # draw origin in axis coordinates (which coincide with world=0)
    ax.plot(0, 0, "c*", markersize=10, label="origin")

    # # draw target  position
    targ_axis = pixel_to_axis(to_pixel_coords(positions["target"]))
    ax.plot(*targ_axis, "go", markersize=8, label="target")

    # helper to draw a single arrow (world orientation) at a pose
    def _draw_arrow(pose: Pose2D, color: str, label: str = None):
        px, py = pixel_to_axis(to_pixel_coords(pose))
        theta = pose.theta
        SIZE = 0.8
        length = SIZE * 1.2
        dx = np.cos(theta) * length
        dy = np.sin(theta) * length
        # pass label only if provided (None is ignored by matplotlib)
        ax.arrow(
            px,
            py,
            dx,
            dy,
            head_width=SIZE,
            head_length=SIZE,
            length_includes_head=True,
            fc=color,
            ec=color,
            label=label,
        )

    # draw our car in red (label only once)
    _draw_arrow(positions["seti"], "r", "SETI")
    # draw opponents in blue; only the first gets a legend entry
    for idx, opponent in enumerate(positions["opponents"]):
        lbl = (
            "opponent" + f"{'s' if len(positions['opponents']) > 1 else ''}"
            if idx == 0
            else None
        )
        _draw_arrow(opponent, "b", lbl)

    # show legend for all labeled artists
    ax.legend(loc="upper right")

    ax.set_title("Positions on inflated occupancy Grid")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.axis("off")


fig_gen.add_figure("Positions", draw_positions)


# ====== Run A* on opponents ======
# Get path length
def get_path_length(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    length = 0.0
    for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
        length += np.hypot(x2 - x1, y2 - y1)
    return length / FACTOR_PIXEL_PER_METER  # convert back to meters


# Use nx.astar_path()
# Build graph from occupancy grid where each free cell is a node and edges connect 8-connected neighbors
def build_graph_from_occupancy_grid(
    occupancy_grid: NDArray, enable_diagonal: bool = True
) -> Graph:
    G = Graph()
    h, w = occupancy_grid.shape
    # we'll allow 4-connected moves always, plus diagonals only when the
    # two touching orthogonal neighbors are both free (no wall-cutting).
    cardinal = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    diagonal = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    for y in range(h):
        for x in range(w):
            if not occupancy_grid[y, x]:  # free cell
                G.add_node((x, y))
                # cardinal neighbors
                for dx, dy in cardinal:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not occupancy_grid[ny, nx]:
                        G.add_edge((x, y), (nx, ny), weight=1.0)
                # diagonal neighbors
                if enable_diagonal:
                    for dx, dy in diagonal:
                        nx, ny = x + dx, y + dy
                        # require that both adjacent cardinal cells are free
                        adj1 = (x + dx, y)  # horizontal step
                        adj2 = (x, y + dy)  # vertical step
                        if (
                            0 <= nx < w
                            and 0 <= ny < h
                            and not occupancy_grid[ny, nx]
                            and 0 <= adj1[0] < w
                            and 0 <= adj1[1] < h
                            and not occupancy_grid[adj1[1], adj1[0]]
                            and 0 <= adj2[0] < w
                            and 0 <= adj2[1] < h
                            and not occupancy_grid[adj2[1], adj2[0]]
                        ):
                            G.add_edge((x, y), (nx, ny), weight=np.sqrt(2))
    return G


graph = build_graph_from_occupancy_grid(
    occupancy_grid, enable_diagonal=ASTAR_DO_DIAGONAL
)


# Convert world pose (metres) to pixel coordinates used by the graph.
# Returned tuple is in image pixel space with origin at top-left corner.
def to_pixel_coords(pose: Pose2D) -> tuple[int, int]:
    x = int(pose.x * FACTOR_PIXEL_PER_METER + w / 2)
    y = int(pose.y * FACTOR_PIXEL_PER_METER + h / 2)
    return (x, y)


# Convert image pixel coords to the axis coordinate system used for all
# plots.  The occupancy grid is displayed with `extent=(-w/2,w/2,-h/2,h/2)`
# and `origin='lower'`, so pixel (0,0) maps to (-w/2,-h/2) in axis units.
def pixel_to_axis(pixel: tuple[int, int]) -> tuple[float, float]:
    px, py = pixel
    return px - w / 2, py - h / 2


def compute_astar(
    start_pixel: tuple[int, int], goal_pixel: tuple[int, int]
) -> list[tuple[int, int]]:
    start = start_pixel
    goal = goal_pixel
    print(f"computing path from {start} to {goal}")
    try:
        p = nx.astar_path(graph, start, goal)
        print(f"found path with {len(p)} nodes")
        return p
    except nx.NetworkXNoPath:
        print(f"no path between {start} and {goal}")
        return []


def make_draw_path(path: list[tuple[int, int]], name: str) -> Callable[[Axes], None]:
    def draw(ax: Axes, name=name, path=path):
        h_, w_ = occupancy_grid.shape
        extent = (-w_ / 2, w_ / 2, -h_ / 2, h_ / 2)
        ax.imshow(
            np.bitwise_not(occupancy_grid), cmap="gray", extent=extent, origin="lower"
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        if path:
            path_length = get_path_length(path)
            path_axis = [(x - w_ / 2, y - h_ / 2) for x, y in path]
            pxs, pys = zip(*path_axis)
            ax.plot(pxs, pys, "m.", linewidth=2, label="A* Path")
            sx, sy = path_axis[0]
            gx, gy = path_axis[-1]
            ax.plot([sx], [sy], "ro", label="start")
            ax.plot([gx], [gy], "go", label="goal")
            ax.set_title(f"A* for {name} (length: {path_length:.2f} m)")
            ax.legend()
        else:
            ax.set_title("A* Path (none found)")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.axis("off")

    return draw


# compute A* for opponents
# all_starts = [("seti", positions["seti"])]
all_starts = []
for i, opp in enumerate(positions["opponents"], start=1):
    all_starts.append((f"opp{i}", opp))

paths = {}
goal_pixel = to_pixel_coords(positions["target"])
for name, pose in all_starts:
    sp = to_pixel_coords(pose)
    print(f"start pixel {sp} for {name}")
    path = compute_astar(sp, goal_pixel)
    paths[name] = path
    fig_gen.add_figure(f"A* Path {name}", make_draw_path(path, name))

# ====== Position prediction ======

estimated_positions: dict[str, list[Pose2D]] = {}

for name, path in paths.items():
    length = get_path_length(path)
    print(f"Path length for {name}: {length:.2f} m")

    # Estimate positions along the path assuming constant speed equal to ASTAR_AVERAGE_SPEED_MPS
    estimated_positions[name] = []
    if length > 0:
        total_time = length / ASTAR_AVERAGE_SPEED_MPS
        print(f"Estimated time for {name} to reach target: {total_time:.2f} s")
        num_estimates = int(np.ceil(total_time / SAMPLE_INTERVAL_S))
        print(f"Number of position estimates for {name}: {num_estimates}")
        for i in range(num_estimates):
            t = i * SAMPLE_INTERVAL_S
            dist_along_path = min(t * ASTAR_AVERAGE_SPEED_MPS, length)
            # find the segment of the path corresponding to this distance
            dist_accum = 0.0
            for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
                # segment length in pixels
                dist_segment_px = np.hypot(x2 - x1, y2 - y1)
                # convert to meters for comparison with dist_along_path
                dist_segment = dist_segment_px / FACTOR_PIXEL_PER_METER
                if dist_accum + dist_segment >= dist_along_path:
                    # this is the segment where the estimated position lies
                    ratio = (
                        (dist_along_path - dist_accum) / dist_segment
                        if dist_segment > 0
                        else 0
                    )
                    est_x = x1 + ratio * (x2 - x1)
                    est_y = y1 + ratio * (y2 - y1)
                    # convert back to world coordinates
                    est_world_x = (est_x - w / 2) / FACTOR_PIXEL_PER_METER
                    est_world_y = (est_y - h / 2) / FACTOR_PIXEL_PER_METER
                    estimated_positions[name].append(
                        Pose2D(
                            x=est_world_x, y=est_world_y, theta=0.0
                        )
                    )
                    break
                dist_accum += dist_segment

# Show estimated positions on a plot
fig_est = Figure()
ax_est = fig_est.add_subplot(111)
# draw base occupancy grid and vehicles (in pixel/axis coords)
draw_positions(ax_est)
# estimated_positions are stored in world metres; need to convert to
# the axis coordinate system (which is pixel units centered at 0)
# simplest conversion: world * FACTOR_PIXEL_PER_METER
for name, poses in estimated_positions.items():
    if poses:
        xs, ys = zip(
            *[
                (p.x * FACTOR_PIXEL_PER_METER, p.y * FACTOR_PIXEL_PER_METER)
                for p in poses
            ]
        )
        ax_est.plot(xs, ys, ".--", label=f"Estimated {name} path")
ax_est.set_title(
    f"Estimated Positions Along A* Paths ({ASTAR_AVERAGE_SPEED_MPS} m/s, {1/SAMPLE_INTERVAL_S:.1f} Hz)"
)
ax_est.legend()
fig_gen.add_figure("Estimated Positions", fig_est)


# ====== Path planning procedure ======
# ===== Local commands =====
# Possible commands
class Command(TypedDict):
    steering_angle: float  # rad
    speed: float  # m/s

COMMANDS: list[Command] = [
    # {"steering_angle": 0.0, "speed": 1.0},  # straight
    {"steering_angle": 0.0, "speed": 0.3},  # straight slow
    {"steering_angle": np.radians(60), "speed": 0.3}, # sharp left
    {"steering_angle": np.radians(-60), "speed": 0.3},# sharp right
    # {"steering_angle": np.radians(15), "speed": 0.3}, # left
    # {"steering_angle": np.radians(-15), "speed": 0.3},# right
]

def get_commands_copy() -> list[Command]:
    """Return all possible commands (copy)."""
    return [Command(steering_angle=cmd["steering_angle"], speed=cmd["speed"]) for cmd in COMMANDS]

# storage for precomputed trajectories (local coordinate frames)
COMMANDS_PREDICTED_TRAJECTORY_POSITIONS: list[list[Pose2D]] = []

# helpers for command trajectories
def get_trajectory_radius(steering_angle: float) -> float:
    if steering_angle == 0.0:
        return float("inf")  # straight line
    return WHEELBASE_M / np.tan(steering_angle)


def get_command_trajectory(command: Command) -> list[Pose2D]:
    """Return a list of poses (x,y,theta) in local frame for the given command.
    Samples are taken every ``SAMPLE_INTERVAL_S`` seconds over
    ``PATH_PLANNING_HORIZON_S`` seconds. Distance along path = speed * time.
    The theta value corresponds to the turning angle (zero for straight motion).
    """
    radius = get_trajectory_radius(command["steering_angle"])
    speed = command["speed"]

    # number of sampling instants (including t=0)
    n_steps = int(np.ceil(SAMPLE_INTERVAL_S / SAMPLE_INTERVAL_S)) + 1
    n_steps = max(n_steps, 2)

    poses: list[Pose2D] = []
    for i in range(n_steps):
        t = i * SAMPLE_INTERVAL_S
        dist = speed * t
        if radius == float("inf"):
            # Straight ahead along +X (theta=0 convention)
            x = dist
            y = 0.0
            theta = 0.0
        else:
            ang = dist / abs(radius)
            # CCW-positive: left turn (radius>0) gives positive theta,
            # right turn (radius<0) gives negative theta.
            x = abs(radius) * np.sin(ang)
            y = radius * (1 - np.cos(ang))
            theta = np.sign(radius) * ang
        poses.append(Pose2D(x=x, y=y, theta=theta))
    return poses


def draw_trajectory(ax: Axes, name: str, index: int):
    """Plot the precomputed trajectory for command identified by ``name``.
    In addition to connecting the sample points, draw small arrows at each
    pose to indicate the local heading (theta)."""
    poses = COMMANDS_PREDICTED_TRAJECTORY_POSITIONS[index]
    if poses:
        # local-frame points are computed with forward = +X; for the
        # display the user wants the vehicle moving upward (toward +Y).
        # rotate all plotted data by +90° CCW: (x,y) -> (-y,x) and
        # shift heading by +π/2.
        xs = [p.x for p in poses]
        ys = [p.y for p in poses]
        thetas = [p.theta for p in poses]
        xs_plot = [-y for y in ys]
        ys_plot = xs
        thetas_plot = [(t + np.pi / 2) % (2 * np.pi) for t in thetas]
        ax.plot(xs_plot, ys_plot, "x--", label=name)
        # draw tiny orientation tips; CCW-positive: cos→dx, sin→dy
        SIZE = 0.005
        for x, y, theta in zip(xs_plot, ys_plot, thetas_plot):
            dx = np.cos(theta) * SIZE * 1.2
            dy = np.sin(theta) * SIZE * 1.2
            ax.arrow(x, y, dx, dy,
                    head_width=SIZE,
                    head_length=SIZE,
                    length_includes_head=True,
                     fc='k', ec='k')
    ax.set_title("Predicted Trajectories for Possible Commands")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.set_aspect("auto")
    ax.grid(True)


fig_traj = Figure()
ax_traj = fig_traj.add_subplot(111)
# compute and plot each command's trajectory
for idx, cmd in enumerate(COMMANDS):
    name = f"Command {idx + 1}"
    COMMANDS_PREDICTED_TRAJECTORY_POSITIONS.append(get_command_trajectory(cmd))
    draw_trajectory(ax_traj, name, idx)
fig_gen.add_figure("Predicted Trajectories", fig_traj)

# ===== Path planning =====

class TrajectoryPoint(TypedDict):
    pose: Pose2D # Pose at this point
    command: Command | None # Command chosen to reach this point from the previous one (None for the first point)
    remaining_commands: list[Command]

trajectory_plan_length = int(np.ceil(PATH_PLANNING_HORIZON_S / SAMPLE_INTERVAL_S)) # Number of points to have
# trajectory_plan_length = 6
trajectory: list[TrajectoryPoint] = [
    # Start with position of seti
    TrajectoryPoint(
        pose=positions["seti"],
        command=None,
        remaining_commands=get_commands_copy()
    )
]

iter_counter = 0
print(f"Starting path planning procedure for {trajectory_plan_length} points...")
while len(trajectory) < trajectory_plan_length:
    iter_counter += 1
    if (iter_counter % 100_000) == 0:
        print(f"Iteration {iter_counter}, tl={len(trajectory)}, trl={len(trajectory[-1]['remaining_commands'])}, c={trajectory[-1]['command']}")
    
    # Procedure
    current_point = trajectory[-1]
    if not current_point["remaining_commands"]: # No remaining commands from this point
        if current_point["command"] is None: # Initial position, exhausted all commands, can't do anything
            break
        else: # Backtrack to previous point and try a different command
            trajectory.pop() # Remove current point (backtracking, should always have a previous point since initial point has all commands)
            continue

    # Get next command to try
    # next_command = current_point["remaining_commands"].pop(0) # Get and remove the first remaining command
    next_command_index = np.random.choice(len(current_point["remaining_commands"]))
    next_command = current_point["remaining_commands"].pop(next_command_index)
    
    # Get trajectory for this command
    cmd_traj = get_command_trajectory(next_command)
    # Get trajectory points in global coordinates by applying the local trajectory to the current point's pose
    global_traj = [p.to_global(current_point["pose"]) for p in cmd_traj]
    # Check if any point in the global trajectory collides with an obstacle in the occupancy grid
    collision = False
    # Collision with occupancy grid if any point in global_traj is in an occupied cell
    for p in global_traj:
        px, py = to_pixel_coords(p)
        if px < 0 or px >= w or py < 0 or py >= h:
            collision = True
            break
        elif occupancy_grid[py, px]:
            collision = True
            break

    if collision:
        continue # Try the next command

    # No collision, add the last point of this trajectory to the plan and continue
    trajectory.append(TrajectoryPoint(
        pose=global_traj[-1],
        command=next_command,
        remaining_commands=get_commands_copy() # reset commands for the new point
    ))
print(f"Planned trajectory with {len(trajectory)} points!")

# Draw the planned trajectory on the occupancy grid, show triangles at each point indicating the heading, and annotate with the command used to get there.
fig_plan = Figure()
ax_plan = fig_plan.add_subplot(111)

# draw occupancy grid as background so arrows/trajectory sit on top
h_, w_ = occupancy_grid.shape
extent_plan = (-w_/2, w_/2, -h_/2, h_/2)
ax_plan.imshow(np.bitwise_not(occupancy_grid), cmap='gray', extent=extent_plan, origin='lower')
ax_plan.set_xlim(extent_plan[0], extent_plan[1])
ax_plan.set_ylim(extent_plan[2], extent_plan[3])

# convert world poses to axis coords (matching occupancy grid transform)
axis_coords = [pixel_to_axis(to_pixel_coords(p["pose"])) for p in trajectory]
xs, ys = zip(*axis_coords) if axis_coords else ([], [])
thetas = [p["pose"].theta for p in trajectory]  # heading unaffected by conversion

ax_plan.plot(xs, ys, "--", label="Planned Trajectory")

# Draw orientation arrows using same style as opponents (blue)
# CCW-positive: cos→dx, sin→dy
ARROW_SIZE = 1  # metres
for x, y, theta in zip(xs, ys, thetas):
    dx = np.cos(theta) * ARROW_SIZE * 1.2
    dy = np.sin(theta) * ARROW_SIZE * 1.2
    ax_plan.arrow(x, y, dx, dy,
                  head_width=ARROW_SIZE,
                  head_length=ARROW_SIZE,
                  length_includes_head=True,
                  fc='b', ec='b')

# Annotate with commands (if available)
for i, point in enumerate(trajectory):
    if point["command"]:
        index_cmd = COMMANDS.index(point["command"])
        ax_plan.annotate(f"i={i}, (com:{index_cmd + 1})", (xs[i], ys[i]), xytext=(5, 5), textcoords="offset points")

ax_plan.set_title("Planned Trajectory")
ax_plan.set_xlabel("X (m)")
ax_plan.set_ylabel("Y (m)")
ax_plan.legend()
ax_plan.grid(True)
fig_gen.add_figure("Planned Trajectory", fig_plan)


# ====== Show figures ======

# Show figures

fig_gen.show()
