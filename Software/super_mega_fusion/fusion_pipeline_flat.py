"""
Pipeline déroulée au plus simple
"""
from pathlib import Path
from typing import Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from numpy.typing import NDArray

# =======================================
#  Parameters
# =======================================

CAM_BAND_Y_MIN = 20
CAM_BAND_Y_MAX = 25
CAM_FOV_DEG = np.rad2deg(0.785398) # Field of view of the camera (degrees) 
LIDAR_FOV_DEG = 360 # Field of view of the LIDAR (degrees), centered on the front of the car, so e.g. 270 means 135 degrees to the left and 135 degrees to the right. Note: this is not necessarily the same as the camera FoV, which may be wider or narrower.


# HSV ranges
WALL_RED_HSV_MIN = (162/179, 200/255, 15/255)
WALL_RED_HSV_MAX = (12/179, 1.0, 1.0)
WALL_GREEN_HSV_MIN = (33/179, 157/255, 47/255)
WALL_GREEN_HSV_MAX = (72/179, 1.0, 1.0)

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
                ax.axis('off')
            else:
                ax.imshow(item)
                ax.axis('off')
        plt.tight_layout()
        plt.show()

def get_camera_frame() -> NDArray[np.uint8]: # As RGB array
    # Load camera_capture.png
    # image_path = Path(__file__).parent / "test.png"
    image_path = Path(__file__).parent / "camera_capture.png"
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

# def get_lidar_ranges() -> NDArray[np.float32]:
#     # Simulate LIDAR data as a 1D array of distances (in meters)
#     # Use sine
#     angles = np.linspace(-LIDAR_FOV_DEG / 2, LIDAR_FOV_DEG / 2, 61) # From -FOV/2 to FOV/2, with 1 degree resolution
#     distances = 2.5 + 1.5 * np.sin(np.radians(angles)*4) # Simulate some variation in distances
#     return distances.astype(np.float32)

def get_lidar_ranges() -> NDArray[np.float32]:
    # Load lidar_scan.txt
    lidar_path = Path(__file__).parent / "lidar_scan.txt"
    lidar_ranges = np.loadtxt(lidar_path, dtype=np.float32)
    if False: # Set nan values to 10m
        lidar_ranges = np.where(np.isnan(lidar_ranges), 10.0, lidar_ranges)
    return lidar_ranges

def get_mask(image_hsv_norm: NDArray[np.float32], hsv_min: tuple[float, float, float], hsv_max: tuple[float, float, float]) -> NDArray[np.bool_]:
    """Return a boolean mask where pixels fall inside the given HSV range.

    - `hsv_min` / `hsv_max` are (H, S, V) with values in [0, 1].
    - Hue wrap-around is supported (e.g. hl > hh means H >= hl or H <= hh).
    """
    h, s, v = image_hsv_norm[..., 0], image_hsv_norm[..., 1], image_hsv_norm[..., 2]
    hl, sl, vl = hsv_min
    hh, sh, vh = hsv_max

    mask_h = ((h >= hl) & (h <= hh)) if hl <= hh else ((h >= hl) | (h <= hh))
    mask = mask_h & (s >= sl) & (s <= sh) & (v >= vl) & (v <= vh)
    return mask

def clustering(points: np.ndarray, *, eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
    """Cluster a set of points and return labels as ``np.uint8``.

    Behavior
    - Uses ``sklearn.cluster.DBSCAN`` when available.
    - Falls back to a simple NumPy connected-component algorithm (distance <= eps)
      when scikit-learn is not installed.
    - Noise points are labeled ``255`` (uint8). Valid cluster ids are 0..254.

    Parameters
    - points: (N, D) array-like of float coordinates.
    - eps: maximum distance between samples in the same neighborhood.
    - min_samples: minimum points to form a core cluster (same meaning as DBSCAN).

    Returns
    - labels: (N,) uint8 array. 255 means noise / unclustered.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0,), dtype=np.uint8)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)

    n = pts.shape[0]

    # Try scikit-learn DBSCAN first (import locally to avoid hard dependency)
    try:
        from sklearn.cluster import DBSCAN

        db = DBSCAN(eps=eps, min_samples=min_samples)
        db.fit(pts)
        labels = db.labels_.astype(int)  # -1 for noise
    except Exception:
        # Fallback: build adjacency (distance <= eps) and extract connected components
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        adj = dists <= eps
        labels = -np.ones(n, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0
        for i in range(n):
            if visited[i]:
                continue
            # BFS/DFS to collect connected component
            stack = [i]
            comp = []
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                nbrs = np.nonzero(adj[u])[0]
                for v in nbrs:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            if len(comp) >= min_samples:
                for idx in comp:
                    labels[idx] = cluster_id
                cluster_id += 1
            # else leave as -1 (noise)

    # Map -1 -> 255 and ensure dtype uint8. Compress labels if there are >254 clusters.
    unique_pos = np.unique(labels[labels >= 0])
    if unique_pos.size > 254:
        # compress mapping to 0..254
        mapping = {old: new for new, old in enumerate(unique_pos[:254])}
        out = np.full(n, 255, dtype=np.uint8)
        for i in range(n):
            if labels[i] in mapping:
                out[i] = mapping[labels[i]]
    else:
        out = np.where(labels == -1, 255, labels).astype(np.uint8)

    return out
    

# =======================================
#  Main
# =======================================

fig_gen = FigureGenerator()

# ============= From Camera ==============
# Get image
camera_frame = get_camera_frame()
fig_gen.add_figure("Camera Frame", camera_frame)

# Extract band at center of image
# height, width, _ = camera_frame.shape
camera_band = camera_frame[CAM_BAND_Y_MIN:CAM_BAND_Y_MAX, :, :]
fig_gen.add_figure("Camera Band", camera_band)

# ============= From LIDAR ==============
# Get LIDAR data as a 1D array of distances (in meters)
lidar_ranges = get_lidar_ranges()
# Make figure: funciton of angle (x-axis) and distance (y-axis)
lidar_angles = np.linspace(-LIDAR_FOV_DEG / 2, LIDAR_FOV_DEG / 2, len(lidar_ranges))
lidar_fig = plt.figure(figsize=(10, 5))
ax = lidar_fig.add_subplot(1, 1, 1)
ax.plot(lidar_angles, lidar_ranges, "ro")
fig_gen.add_figure(f"LIDAR Ranges (FoV of {LIDAR_FOV_DEG:.1f}°)", lidar_fig)
plt.close(lidar_fig)  # close pyplot-managed window (we render via Agg)
# Plot 2D top-down view (x = range * sin(angle), y = range * cos(angle))
lidar_x = lidar_ranges * np.sin(np.radians(lidar_angles))
lidar_y = lidar_ranges * np.cos(np.radians(lidar_angles))
lidar_topdown_fig = plt.figure(figsize=(5, 5))
ax = lidar_topdown_fig.add_subplot(1, 1, 1)
ax.scatter(lidar_x, lidar_y, color="red")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("LIDAR Points in Top-Down View")
fig_gen.add_figure("LIDAR Points (Top-Down View)", lidar_topdown_fig)
plt.close(lidar_topdown_fig)

# ============= Extract features ==============
# Convert to HSV and normalize to 0..1 floats
camera_band_hsv = cv2.cvtColor(camera_band, cv2.COLOR_RGB2HSV)
camera_band_hsv_float = camera_band_hsv.astype('float32')
camera_band_hsv_float[:, :, 0] = camera_band_hsv_float[:, :, 0] / 179.0
camera_band_hsv_float[:, :, 1] = camera_band_hsv_float[:, :, 1] / 255.0
camera_band_hsv_float[:, :, 2] = camera_band_hsv_float[:, :, 2] / 255.0

# Create masks
wall_red_mask = get_mask(camera_band_hsv_float, WALL_RED_HSV_MIN, WALL_RED_HSV_MAX)
wall_green_mask = get_mask(camera_band_hsv_float, WALL_GREEN_HSV_MIN, WALL_GREEN_HSV_MAX)

# Show masks
fig_gen.add_figure("Wall Red Mask", wall_red_mask)
fig_gen.add_figure("Wall Green Mask", wall_green_mask)

# Mask combined (ordered by priority)
feature_mask = np.ones(wall_red_mask.shape, dtype=np.uint8) * 255 # Remaining at 255, Enemies
feature_mask[wall_red_mask] = 1 # 1: Red Wall
feature_mask[wall_green_mask] = 2 # 2: Green Wall
# feature_mask[] = 0 # 0: Background

# ============= Split Band into nl segments ==============
# Get nl (Lidar points in camera FoV)
lidar_in_range_mask = (-CAM_FOV_DEG / 2 <= lidar_angles) & (lidar_angles <= CAM_FOV_DEG / 2)
lidar_angles_in_fov = lidar_angles[lidar_in_range_mask]
lidar_ranges_in_fov = lidar_ranges[lidar_in_range_mask]
nl = len(lidar_angles_in_fov)

# Plot
lidar_in_range_fig = plt.figure(figsize=(10, 5))
ax = lidar_in_range_fig.add_subplot(1, 1, 1)
ax.plot(lidar_angles_in_fov, lidar_ranges_in_fov, "ro")
fig_gen.add_figure(f"LIDAR Ranges in Camera FoV ({CAM_FOV_DEG:.1f}°, {len(lidar_angles_in_fov)} points)", lidar_in_range_fig)# DEG with 1 digits after comma
plt.close(lidar_in_range_fig)

# Split band into nl segments
indices = np.linspace(0, camera_band.shape[1], nl + 1, dtype=int)
# Get segment map: uint16 2D array, with number corresponding to segment index (0 to nl-1) for each pixel
segment_map = np.zeros(camera_band.shape[:2], dtype=np.uint16)
for i in range(nl):
    segment_map[:, indices[i]:indices[i + 1]] = i

fig_gen.add_figure("Segment Map", segment_map)

# ===== Helper ====

# Subsubplot: on top, union of masks with different colors; on bottom, range and lidar angle in fov
mask_union_fig = plt.figure(figsize=(10, 5))
ax1 = mask_union_fig.add_subplot(2, 1, 1)
# Create a color image where each feature has a different color
color_image = np.zeros((*feature_mask.shape, 3), dtype=np.uint8)
color_image[feature_mask == 0] = [128, 128, 128] # Gray for Background
color_image[feature_mask == 1] = [255, 0, 0] # Red for Red Wall
color_image[feature_mask == 2] = [0, 255, 0] # Green for Green Wall
color_image[feature_mask == 255] = [0, 0, 255] # Blue for Enemy
ax1.imshow(color_image)
ax1.set_title("Feature Mask Union")
ax1.axis('off')
ax2 = mask_union_fig.add_subplot(2, 1, 2)
ax2.plot(lidar_angles_in_fov, lidar_ranges_in_fov, "ro")
ax2.set_title(f"LIDAR Ranges (FoV of {CAM_FOV_DEG:.1f}°)")
ax2.set_xlabel("Angle (°)")
ax2.set_ylabel("Range (m)")
fig_gen.add_figure("Feature Mask and LIDAR Ranges", mask_union_fig)
plt.close(mask_union_fig)


# ============= Assign features to segments ==============
# For each segment, get majority feature (0=Red Wall, 1=Green Wall, 255=Enemy)
segment_features = np.zeros(nl, dtype=np.uint8)
for i in range(nl):
    segment_mask = (segment_map == i)
    if np.any(segment_mask):
        # Get majority feature in this segment
        features_in_segment = feature_mask[segment_mask]
        majority_feature = np.bincount(features_in_segment).argmax()
        segment_features[i] = majority_feature
    else:
        segment_features[i] = 0 # No pixels in this segment, consider it as Background

feature_names = {0: "Background", 1: "Red Wall", 2: "Green Wall", 255: "Enemy"}
feature_colors = {0: "gray", 1: "red", 2: "green", 255: "blue"}

# ============= Fusion =============
# Assign to each lidar point in camera FoV the feature of its corresponding segment
lidar_features_in_fov = np.zeros_like(lidar_ranges_in_fov, dtype=np.uint8)
for i in range(nl):
    lidar_features_in_fov[i] = segment_features[i]

# Show LIDAR points with feature colors
lidar_feature_fig = plt.figure(figsize=(10, 5))
ax = lidar_feature_fig.add_subplot(1, 1, 1)
for feat_value, feat_name in feature_names.items():
    feat_mask = (lidar_features_in_fov == feat_value)
    ax.scatter(lidar_angles_in_fov[feat_mask], lidar_ranges_in_fov[feat_mask], label=feat_name, color=feature_colors[feat_value])

# Use the actual lidar angles as tick positions so labels line up with plotted points
ax.set_xticks(lidar_angles_in_fov)
ax.set_xticklabels([f"{angle:.1f}°" for angle in lidar_angles_in_fov], rotation=45)
ax.set_xlim(lidar_angles_in_fov[0] - 1e-3, lidar_angles_in_fov[-1] + 1e-3)
ax.set_xlabel("Angle (°)")
ax.set_ylabel("Range (m)")
ax.set_title("LIDAR Points with Fused Features")
ax.legend()
fig_gen.add_figure("LIDAR Points with Fused Features", lidar_feature_fig)
plt.close(lidar_feature_fig)

# ============= Find enemies ==============
# Keep lidar points labeled as "Enemy" (255)
enemy_mask = (lidar_features_in_fov == 255)
enemy_angles = lidar_angles_in_fov[enemy_mask]
enemy_ranges = lidar_ranges_in_fov[enemy_mask]

# Plot in 2D top-down view (x = range * sin(angle), y = range * cos(angle))
enemy_x = enemy_ranges * np.sin(np.radians(enemy_angles))
enemy_y = enemy_ranges * np.cos(np.radians(enemy_angles))
enemy_fig = plt.figure(figsize=(5, 5))
ax = enemy_fig.add_subplot(1, 1, 1)
ax.scatter(enemy_x, enemy_y, color="blue", label="Enemy")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Detected Enemies in Top-Down View")
ax.legend()
fig_gen.add_figure("Detected Enemies (Top-Down View)", enemy_fig)
plt.close(enemy_fig)

# Clustering enemy points to find distinct enemies
enemy_points = np.column_stack((enemy_x, enemy_y))
enemy_labels = clustering(enemy_points, eps=0.5, min_samples=1)
# Plot clustered enemies with different colors
enemy_cluster_fig = plt.figure(figsize=(5, 5))
ax = enemy_cluster_fig.add_subplot(1, 1, 1)
unique_labels = np.unique(enemy_labels)
for label in unique_labels:
    cluster_mask = (enemy_labels == label)
    ax.scatter(enemy_x[cluster_mask], enemy_y[cluster_mask], label=f"Enemy {label}", s=100)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Clustered Enemies in Top-Down View")
ax.legend()
fig_gen.add_figure("Clustered Enemies (Top-Down View)", enemy_cluster_fig)
plt.close(enemy_cluster_fig)

# Get Enemy clusters' centers
enemy_cluster_centers = np.array([enemy_points[enemy_labels == label].mean(axis=0) for label in unique_labels])

# Show 2D top-down view of all LIDAR points with enemies removed, and add enemies back with different color for each enemy cluster
lidar_no_enemy_fig = plt.figure(figsize=(5, 5))
ax = lidar_no_enemy_fig.add_subplot(1, 1, 1)
# Use in-FoV coordinates so the boolean mask (length `nl`) matches the plotted arrays
non_enemy_mask = (lidar_features_in_fov != 255)
lidar_x_in_fov = lidar_ranges_in_fov * np.sin(np.radians(lidar_angles_in_fov))
lidar_y_in_fov = lidar_ranges_in_fov * np.cos(np.radians(lidar_angles_in_fov))
if np.any(non_enemy_mask):
    ax.scatter(lidar_x_in_fov[non_enemy_mask], lidar_y_in_fov[non_enemy_mask], color="gray", label="Non-Enemy")
# Plot enemy clusters with different colors (enemy_x/ enemy_y are already in‑FoV)
for label in unique_labels:
    cluster_mask = (enemy_labels == label)
    ax.scatter(enemy_x[cluster_mask], enemy_y[cluster_mask], label=f"Enemy {label}", s=100)
# Plot cluster centers
ax.scatter(enemy_cluster_centers[:, 0], enemy_cluster_centers[:, 1], color="blue", marker="x", s=200, label="Enemy Cluster Centers")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
# ax.set_aspect('equal', adjustable='box')
ax.set_title("LIDAR Points with Clustered Enemies (Top-Down View)")
ax.legend()
fig_gen.add_figure("LIDAR with Clustered Enemies (Top-Down View)", lidar_no_enemy_fig)
plt.close(lidar_no_enemy_fig)

fig_gen.show()


