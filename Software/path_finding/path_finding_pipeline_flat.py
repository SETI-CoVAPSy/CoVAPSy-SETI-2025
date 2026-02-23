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
from networkx import Graph, DiGraph
from scipy.ndimage import binary_dilation, distance_transform_edt
from dataclasses import dataclass
from enum import Enum
from matplotlib.patches import Circle

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
    0.3  # assumed average speed of opponents for time estimation (m/s)
)
SAMPLE_INTERVAL_S = 0.3   # s, how often to sample points along A* path / random tree for position estimation
LOCAL_SAMPLE_INTERVAL_S = SAMPLE_INTERVAL_S / 4 # s, time interval between successive points in the local command trajectories (should be <= SAMPLE_INTERVAL_S for consistency)

WHEELBASE_M = 0.5  # m, distance between front and rear axles for trajectory prediction
# old distance-based sampling constant removed; time interval SAMPLE_INTERVAL_S now used
PATH_PLANNING_HORIZON_S = 20.0  # s, how far into the future to plan paths

OPPONENT_COLLISION_RADIUS_M = 0.1  # m, radius around opponent positions to consider as collision for trajectory checking

ASTAR_BIAS_STRENGTH = 15.0  # softmax temperature multiplier for biasing command selection toward A* path
                            # 0 = uniform random, higher = more deterministic toward A* reference

MAX_ITERATIONS = 200_000 # max iterations for path planning procedure (to prevent infinite loops)

FORWARD_DOT_THRESHOLD = 0.0  # minimum dot-product of an edge vector with the local forward direction
                              # to be included in the directed graph.  0.0 = strict forward half-space;
                              # use a small negative value (e.g. -0.3) to allow slight backward lean.

TRACK_FORWARD_REVERSED = True  # False → green wall on right / red on left  (forward = green→red cross-track)
                                 # True  → red wall on right / green on left  (forward reversed)

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
        # "target": Pose2D(x=1.0, y=2.0, theta=0.0),
        "target": Pose2D(x=-1.0, y=-2, theta=0.0),
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

    def show_at_index(self, index: int, blocking: bool = True) -> None:
        """Show only the figure at the given index."""
        index = index % len(self.figures)  # wrap around if out of bounds

        name, item = self.figures[index]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(name)
        if callable(item):
            item(ax)
        elif isinstance(item, Figure):
            ax.imshow(figure_to_array(item))
            ax.axis("off")
        else:
            ax.imshow(item)
            ax.axis("off")
        plt.tight_layout()
        plt.show(block=blocking)

class Occupancy(Enum):
    FREE = 0
    MISC_OBJECT = 1
    WALL_RED = 10
    WALL_GREEN = 11

OccupancyGrid: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]  # 2D uint8 array: 0=FREE, 1=MISC_OBJECT, 10=WALL_RED, 11=WALL_GREEN

def get_occupancy_grid() -> OccupancyGrid:
    """Load carte.png (RGB) and map pixel colours to Occupancy values.

    Colour mapping:
      white (255,255,255) → FREE
      red   (255,  0,  0) → WALL_RED
      green (  0,255,  0) → WALL_GREEN
      any other colour    → MISC_OBJECT
    """
    img = Image.open(Path(__file__).parent / "carte.png").convert("RGB")
    img_array = np.array(img, dtype=np.uint8)   # shape (H, W, 3)
    img_array = np.rot90(img_array, 2)
    img_array = np.flip(img_array, 1)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    result = np.full(r.shape, Occupancy.MISC_OBJECT.value, dtype=np.uint8)
    result[(r == 255) & (g == 255) & (b == 255)] = Occupancy.FREE.value
    result[(r == 255) & (g == 0)   & (b == 0)]   = Occupancy.WALL_RED.value
    result[(r == 0)   & (g == 255) & (b == 0)]   = Occupancy.WALL_GREEN.value
    return result


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
occupancy_grid_original = get_occupancy_grid()
# copy for calculations, will be dilated
occupancy_grid = occupancy_grid_original.copy()

# compute extent in pixel coordinates so that origin (0,0) → center of image
h, w = occupancy_grid.shape
extent = (-w / 2, w / 2, -h / 2, h / 2)

fig_occ = Figure()
ax_occ = fig_occ.add_subplot(111)
# origin='lower' makes y increase upward which matches usual coordinate frames
# draw helper that converts OccupancyGrid values to RGBA colours for display

# Colour palette for each Occupancy value
_OCCUPANCY_COLOURS: dict[int, tuple[int, int, int, int]] = {
    Occupancy.FREE.value:        (255, 255, 255, 255),  # white
    Occupancy.MISC_OBJECT.value: (160, 160, 160, 255),  # gray
    Occupancy.WALL_RED.value:    (220,  50,  50, 255),  # red
    Occupancy.WALL_GREEN.value:  ( 50, 180,  50, 255),  # green
}

def _occupancy_to_rgba(grid: OccupancyGrid) -> np.ndarray:
    """Return an (H, W, 4) uint8 RGBA array for the given OccupancyGrid."""
    h_, w_ = grid.shape
    rgba = np.zeros((h_, w_, 4), dtype=np.uint8)
    for val, colour in _OCCUPANCY_COLOURS.items():
        mask = grid == val
        rgba[mask] = colour
    # unknown values fall back to magenta so they stand out
    known = np.zeros((h_, w_), dtype=bool)
    for val in _OCCUPANCY_COLOURS:
        known |= grid == val
    rgba[~known] = (255, 0, 255, 255)
    return rgba


def _plot_grid(ax):
    ax.imshow(_occupancy_to_rgba(occupancy_grid_original), extent=extent, origin="lower")
    # added pixels mask will be available once dilation has run
    if 'occupancy_dilation_added_pixels' in globals():
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[occupancy_dilation_added_pixels] = (100, 100, 200, 178)  # blue-gray, ~70 % opacity
        ax.imshow(overlay, extent=extent, origin='lower')

_plot_grid(ax_occ)
ax_occ.set_xlim(extent[0], extent[1])
ax_occ.set_ylim(extent[2], extent[3])
fig_gen.add_figure("Occupancy Grid", fig_occ)

# Apply grow operation to inflate obstacles by BINARY_DILATION_PIXELS pixels to account for vehicle size / safety margin.
# The occupancy_grid is a uint8 array, so we dilate on a binary obstacle mask and mark
# newly added pixels as MISC_OBJECT.
_obstacle_mask = occupancy_grid_original != Occupancy.FREE.value
_dilated_mask = binary_dilation(_obstacle_mask, iterations=BINARY_DILATION_PIXELS)
occupancy_dilation_added_pixels = _dilated_mask & ~_obstacle_mask  # bool mask of newly inflated cells
occupancy_grid[occupancy_dilation_added_pixels] = Occupancy.MISC_OBJECT.value

# ====== Forward-direction flow field ======
# Red wall = left side of track, green wall = right side of track when going
# forward.  The cross-track vector pointing green → red is "leftward";
# rotating it 90° clockwise gives the local "forward" direction.
#
# Steps:
#   1. EDT to nearest WALL_RED pixel  (dist_to_red)
#   2. EDT to nearest WALL_GREEN pixel (dist_to_green)
#   3. f_cross = dist_to_green - dist_to_red  (increases toward red = left)
#   4. ∇f_cross points leftward
#   5. forward = rotate ∇f_cross 90° CW:  (gx, gy) → (gy, −gx)
#
# np.gradient(f)[0] = ∂f/∂row = ∂f/∂y_world  (rows align with world y because
# to_pixel_coords maps world y → row index, same direction)
# np.gradient(f)[1] = ∂f/∂col = ∂f/∂x_world

_red_mask_orig   = occupancy_grid_original == Occupancy.WALL_RED.value
_green_mask_orig = occupancy_grid_original == Occupancy.WALL_GREEN.value

_dist_to_red   = distance_transform_edt(~_red_mask_orig).astype(float)
_dist_to_green = distance_transform_edt(~_green_mask_orig).astype(float)

# Cross-track field: increasing toward the "left" wall.  Swapping the sign
# reverses the derived forward direction without touching anything else.
_f_cross = (_dist_to_red - _dist_to_green) if TRACK_FORWARD_REVERSED else (_dist_to_green - _dist_to_red)
_grad_y_fc, _grad_x_fc = np.gradient(_f_cross)   # ∂f/∂y, ∂f/∂x (leftward components)

# Rotate leftward vector 90° CW: (left_x, left_y) → (left_y, −left_x)
forward_field_x: np.ndarray = _grad_y_fc          # shape (H, W)
forward_field_y: np.ndarray = -_grad_x_fc         # shape (H, W)

# Normalise
_fmag = np.hypot(forward_field_x, forward_field_y)
_fmag[_fmag == 0] = 1.0
forward_field_x /= _fmag
forward_field_y /= _fmag

# Visualise the flow field (subsampled quiver on occupancy grid)
def _draw_flow_field(ax):
    _plot_grid(ax)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    N = max(1, min(h, w) // 30)   # subsample step
    ys_q = np.arange(0, h, N)
    xs_q = np.arange(0, w, N)
    Xq, Yq = np.meshgrid(xs_q, ys_q)
    free_q = occupancy_grid[Yq, Xq] == Occupancy.FREE.value
    Xq_ax = Xq[free_q].astype(float) - w / 2
    Yq_ax = Yq[free_q].astype(float) - h / 2
    Uq = forward_field_x[Yq[free_q], Xq[free_q]]
    Vq = forward_field_y[Yq[free_q], Xq[free_q]]
    ax.quiver(Xq_ax, Yq_ax, Uq, Vq,
              scale=50, width=0.003, units='xy', angles='xy',
              color='navy', alpha=0.7)
    ax.set_title("Track Forward Flow Field")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.axis("off")

fig_gen.add_figure("Flow Field", _draw_flow_field)

# Get positions
positions = get_positions()


# Plot positions on occupancy grid
# Draw each vehicle as an arrow oriented by its theta field
def draw_positions(ax: Axes):
    # redraw occupancy grid using helper (original+inflated pixels overlay)
    _plot_grid(ax)
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
# Build a *directed* graph from the occupancy grid.
# When a flow field (fwd_x / fwd_y arrays) is supplied, every potential edge
# A→B is only added when dot(B−A, forward_at_A) ≥ dot_threshold, so A* is
# constrained to travel in the "forward" direction around the circuit.
def build_graph_from_occupancy_grid(
    occupancy_grid: OccupancyGrid,
    enable_diagonal: bool = True,
    fwd_x: np.ndarray | None = None,
    fwd_y: np.ndarray | None = None,
    dot_threshold: float = 0.0,
) -> DiGraph:
    """Return a DiGraph over free pixels.

    If ``fwd_x`` / ``fwd_y`` are provided (normalised forward-direction arrays
    with the same shape as ``occupancy_grid``), only directed edges whose
    direction satisfies dot(edge, forward_at_source) >= ``dot_threshold`` are
    added, enforcing forward-only motion around the track.
    """
    G: DiGraph = DiGraph()
    h, w = occupancy_grid.shape
    use_flow = fwd_x is not None and fwd_y is not None
    cardinal = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    diagonal = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    FREE = Occupancy.FREE.value
    for y in range(h):
        for x in range(w):
            if occupancy_grid[y, x] != FREE:
                continue
            G.add_node((x, y))

            # local forward direction at this source pixel
            fx = float(fwd_x[y, x]) if use_flow else 0.0
            fy = float(fwd_y[y, x]) if use_flow else 0.0

            def _forward_ok(ddx: int, ddy: int) -> bool:
                if not use_flow:
                    return True
                return ddx * fx + ddy * fy >= dot_threshold

            # cardinal neighbors
            for dx, dy in cardinal:
                nx_, ny_ = x + dx, y + dy
                if (
                    0 <= nx_ < w
                    and 0 <= ny_ < h
                    and occupancy_grid[ny_, nx_] == FREE
                    and _forward_ok(dx, dy)
                ):
                    G.add_edge((x, y), (nx_, ny_), weight=1.0)

            # diagonal neighbors (only when both touching cardinal cells are free)
            if enable_diagonal:
                for dx, dy in diagonal:
                    nx_, ny_ = x + dx, y + dy
                    adj1 = (x + dx, y)
                    adj2 = (x, y + dy)
                    if (
                        0 <= nx_ < w
                        and 0 <= ny_ < h
                        and occupancy_grid[ny_, nx_] == FREE
                        and 0 <= adj1[0] < w
                        and 0 <= adj1[1] < h
                        and occupancy_grid[adj1[1], adj1[0]] == FREE
                        and 0 <= adj2[0] < w
                        and 0 <= adj2[1] < h
                        and occupancy_grid[adj2[1], adj2[0]] == FREE
                        and _forward_ok(dx, dy)
                    ):
                        G.add_edge((x, y), (nx_, ny_), weight=np.sqrt(2))
    return G


graph = build_graph_from_occupancy_grid(
    occupancy_grid,
    enable_diagonal=ASTAR_DO_DIAGONAL,
    fwd_x=forward_field_x,
    fwd_y=forward_field_y,
    dot_threshold=FORWARD_DOT_THRESHOLD,
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
        # use helper to draw grid with dilation overlay
        _plot_grid(ax)
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
all_starts = [("seti", positions["seti"])]
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
    {"steering_angle": 0.0, "speed": 1.0},  # straight
    {"steering_angle": 0.0, "speed": 0.3},  # straight slow
    {"steering_angle": np.radians(30), "speed": 0.3}, # sharp left
    {"steering_angle": np.radians(-30), "speed": 0.3},# sharp right
    {"steering_angle": np.radians(60), "speed": 0.3}, # sharp left
    {"steering_angle": np.radians(-60), "speed": 0.3},# sharp right
    {"steering_angle": np.radians(15), "speed": 0.3}, # left
    {"steering_angle": np.radians(-15), "speed": 0.3},# right
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
    n_steps = int(np.ceil(SAMPLE_INTERVAL_S / LOCAL_SAMPLE_INTERVAL_S)) + 1
    n_steps = max(n_steps, 2)

    poses: list[Pose2D] = []
    for i in range(n_steps):
        t = i * LOCAL_SAMPLE_INTERVAL_S
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

# ===== Path planning core =====
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
def clone_trajectory(traj: list[TrajectoryPoint]) -> list[TrajectoryPoint]:
    return [TrajectoryPoint(
        pose=Pose2D(p["pose"].x, p["pose"].y, p["pose"].theta),
        command=p["command"],
        remaining_commands=get_commands_copy()
    ) for p in traj]
best_trajectory: list[TrajectoryPoint] = clone_trajectory(trajectory) # copy of the best trajectory found so far

iter_counter = 0
print(f"Starting path planning procedure for {trajectory_plan_length} points...")
while len(trajectory) < trajectory_plan_length and iter_counter < MAX_ITERATIONS:
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

    # Get next command to try — biased toward the A* reference trajectory
    seti_ref = estimated_positions.get("seti", [])
    step_idx = len(trajectory)  # the time step we are planning for
    n_remaining = len(current_point["remaining_commands"])
    if seti_ref and step_idx < len(seti_ref) and ASTAR_BIAS_STRENGTH > 0 and n_remaining > 1:
        ref_pose = seti_ref[step_idx]
        # score each remaining command by negative distance of its endpoint to the A* reference
        scores = np.empty(n_remaining)
        for ci, cmd in enumerate(current_point["remaining_commands"]):
            # find command index in master list to reuse precomputed trajectory
            try:
                cmd_idx = COMMANDS.index(cmd)
            except ValueError:
                # should not happen, fallback to computing
                local_last = get_command_trajectory(cmd)[-1]
            else:
                local_last = COMMANDS_PREDICTED_TRAJECTORY_POSITIONS[cmd_idx][-1]
            ep = local_last.to_global(current_point["pose"])
            scores[ci] = -np.hypot(ep.x - ref_pose.x, ep.y - ref_pose.y)
        # softmax with temperature = 1 / ASTAR_BIAS_STRENGTH
        # normalize scores to [0,1] so the strength has consistent effect
        # regardless of the absolute distance scale
        score_range = scores.max() - scores.min()
        if score_range > 0:
            scores = (scores - scores.min()) / score_range  # now in [0,1]
        scores *= ASTAR_BIAS_STRENGTH
        scores -= scores.max()  # numerical stability
        weights = np.exp(scores)
        weights /= weights.sum()
        next_command_index = int(np.random.choice(n_remaining, p=weights))
    else:
        next_command_index = int(np.random.choice(n_remaining))
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
        elif occupancy_grid[py, px] != Occupancy.FREE.value:
            collision = True
            break
    if collision:
        continue # Try the next command

    # Collision with opponent planned paths (radius check against all estimated opponent positions)
    # Use estimated_positions at current length of trajectory as index
    for name, opp_poses in estimated_positions.items():
        if name == "seti": # skip our own trajectory
            continue
        if len(opp_poses) > len(trajectory):
            opp_pose = opp_poses[len(trajectory)] # opponent pose at the time we would reach this point
            for p in global_traj:
                dist = np.hypot(p.x - opp_pose.x, p.y - opp_pose.y)
                if dist < OPPONENT_COLLISION_RADIUS_M: # collision radius of 0.5 m
                    collision = True
                    break
        if collision:
            break
    if collision:
        continue # Try the next command

    # No collision, add the last point of this trajectory to the plan and continue
    trajectory.append(TrajectoryPoint(
        pose=global_traj[-1],
        command=next_command,
        remaining_commands=get_commands_copy() # reset commands for the new point
    ))
    if len(trajectory) > len(best_trajectory):
        best_trajectory = clone_trajectory(trajectory) # update best trajectory found so far
print(f"Planned trajectory with {len(best_trajectory)} points!")

# Draw the planned trajectory on the occupancy grid, show triangles at each point indicating the heading, and annotate with the command used to get there.
fig_plan = Figure()
ax_plan = fig_plan.add_subplot(111)

# draw occupancy grid as background so arrows/trajectory sit on top
h_, w_ = occupancy_grid.shape
extent_plan = (-w_/2, w_/2, -h_/2, h_/2)
# reuse helper (which refers to global extent variables; adjust temporarily)
old_extent = extent
extent = extent_plan
_plot_grid(ax_plan)
extent = old_extent
ax_plan.set_xlim(extent_plan[0], extent_plan[1])
ax_plan.set_ylim(extent_plan[2], extent_plan[3])

# also add the A* path for seti (if available) to provide a reference
if "seti" in paths and paths["seti"]:
    seti_axis = [(x - w / 2, y - h / 2) for x, y in paths["seti"]]
    sx, sy = zip(*seti_axis)
    ax_plan.plot(sx, sy, "r-", linewidth=1, markersize=4, label="SETI A* path", alpha=0.3)

# convert world poses to axis coords (matching occupancy grid transform)
axis_coords = [pixel_to_axis(to_pixel_coords(p["pose"])) for p in best_trajectory]
xs, ys = zip(*axis_coords) if axis_coords else ([], [])
thetas = [p["pose"].theta for p in best_trajectory]  # heading unaffected by conversion

# prepare a colour map over the plan length and any opponent estimates
max_op_steps = 0
for poses in estimated_positions.values():
    max_op_steps = max(max_op_steps, len(poses))

n_steps = max(len(best_trajectory), max_op_steps)
if n_steps > 1:
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=n_steps - 1)
    colors = [cmap(norm(i)) for i in range(n_steps)]
else:
    colors = [plt.get_cmap("viridis")(0)]

# plot our car's trajectory using small arrows oriented by heading
# the arrows will also use the time-based colour gradient
if xs and ys:
    U = [np.cos(t) for t in thetas]
    V = [np.sin(t) for t in thetas]
    # use scalar values 0..len(xs)-1 and supply cmap+norm to quiver so
    # the colour gradient is applied correctly to each arrow
    scalars = np.arange(len(xs))
    ax_plan.quiver(xs, ys, U, V, scalars,
                   cmap=cmap, norm=norm,
                   scale=20, width=0.005, units="xy", angles="xy",
                   label="SETI trajectory")
# optionally connect with a faint line for context
ax_plan.plot(xs, ys, "--", color="gray", linewidth=0.5)

# Draw orientation arrows; color them according to time-step gradient
# (may overlap with quiver but provides larger, filled triangles)
ARROW_SIZE = 1  # metres
for i, (x, y, theta) in enumerate(zip(xs, ys, thetas)):
    dx = np.cos(theta) * ARROW_SIZE * 1.2
    dy = np.sin(theta) * ARROW_SIZE * 1.2
    col = colors[i] if i < len(colors) else colors[-1]
    ax_plan.arrow(x, y, dx, dy,
                  head_width=ARROW_SIZE,
                  head_length=ARROW_SIZE,
                  length_includes_head=True,
                  fc=col, ec=col)

# plot opponents' predicted positions using same colour scale, always show all

# convert to axis units (pixels) for plotting
radius_axis = OPPONENT_COLLISION_RADIUS_M * FACTOR_PIXEL_PER_METER
for name, poses in estimated_positions.items():
    if not poses or name == "seti":  # skip if no estimates or if this is our own trajectory
        continue
    for i, p in enumerate(poses):
        ox, oy = pixel_to_axis(to_pixel_coords(p))
        col = colors[min(i, len(colors) - 1)]
        circ = Circle((ox, oy), radius_axis, color=col, alpha=0.1)
        ax_plan.add_patch(circ)
    # add a single legend entry for this opponent series
    ax_plan.scatter([], [], c=[colors[0]], s=100, marker="o",
                    label=f"{name} estimates (radius={OPPONENT_COLLISION_RADIUS_M}m)")

# Annotate with commands (if available)
for i, point in enumerate(best_trajectory):
    if point["command"]:
        index_cmd = COMMANDS.index(point["command"])
        # ax_plan.annotate(f"{i},{index_cmd + 1}", (xs[i], ys[i]), xytext=(5, 5), textcoords="offset points")

ax_plan.set_title("Planned Trajectory")
ax_plan.set_xlabel("X (m)")
ax_plan.set_ylabel("Y (m)")
ax_plan.legend()
ax_plan.grid(True)
fig_gen.add_figure("Planned Trajectory", fig_plan)


# fig_gen.show_at_index(-1, blocking=True)

# ====== Show figures ======

# Show figures

fig_gen.show()
