"""
Camera and LiDAR fusion module for Super Mega Fusion.
"""

import cv2
import numpy as np
from typing import cast
from dataclasses import dataclass
from common import (
    SegmentationLabels,
    ImageLabels,
    ImageArrayHSV,
    ImageArrayRGB,
    ImageArrayMask,
    LidarRanges,
    LidarLabels,
    LidarAngles,
    LidarCartesianPositions,
)
from scipy.spatial.ckdtree import cKDTree


# ====================================================
#  Types
# ====================================================


@dataclass
class HSVMaskRanges:
    h: tuple[float, float]  # [0.0, 179.0]
    s: tuple[float, float]  # [0.0, 255.0]
    v: tuple[float, float]  # [0.0, 255.0]

    def get_mask(self, image_hsv: ImageArrayHSV) -> ImageArrayMask:
        """Get a boolean mask for the given HSV image based on the defined ranges.
        Allow hue range to wrap around the 0-179 boundary."""
        h_min, h_max = self.h
        s_min, s_max = self.s
        v_min, v_max = self.v

        if h_min <= h_max:
            mask_h = (image_hsv[:, :, 0] >= h_min) & (image_hsv[:, :, 0] <= h_max)
        else:
            # Wrap around case
            mask_h = (image_hsv[:, :, 0] >= h_min) | (image_hsv[:, :, 0] <= h_max)

        mask_s = (image_hsv[:, :, 1] >= s_min) & (image_hsv[:, :, 1] <= s_max)
        mask_v = (image_hsv[:, :, 2] >= v_min) & (image_hsv[:, :, 2] <= v_max)

        ret = mask_h & mask_s & mask_v
        return cast(ImageArrayMask, ret)  # Type casting to ImageArrayMask


# ====================================================
#  Functions
# ====================================================


def camera_get_labels(
    image_rgb: ImageArrayRGB,
    cam_band_y_min: int,
    cam_band_y_max: int,
    segmentation_masks: dict[int, HSVMaskRanges],
) -> ImageLabels:
    """Extract labels from given camera image.

    Args:
        image_rgb: Input camera image.
        cam_band_y_min: Minimum y-coordinate of the camera band.
        cam_band_y_max: Maximum y-coordinate of the camera band.
        segmentation_masks: Dictionary of segmentation masks for each label.
    Returns:
        ImageLabels: Labels corresponding to each pixel in the input image."""
    # Extract a band
    image_band_rgb = image_rgb[cam_band_y_min:cam_band_y_max, :, :]

    # Convert RGB to HSV
    image_hsv_ = cv2.cvtColor(image_band_rgb, cv2.COLOR_RGB2HSV)
    image_hsv = cast(ImageArrayHSV, image_hsv_)

    # Get label masks
    mask_wall_red = segmentation_masks[SegmentationLabels.WALL_RED.value].get_mask(
        image_hsv
    )
    mask_wall_green = segmentation_masks[SegmentationLabels.WALL_GREEN.value].get_mask(
        image_hsv
    )

    # Combine masks
    output_labels = (
        np.ones(image_hsv.shape[:2], dtype=np.uint8) * 255
    )  # OPPONENT by default
    output_labels[mask_wall_red] = SegmentationLabels.WALL_RED.value
    output_labels[mask_wall_green] = SegmentationLabels.WALL_GREEN.value

    return output_labels


def lidar_get_labelled_ranges(
    lidar_ranges: LidarRanges,
    camera_labels: ImageLabels,
    lidar_fov_deg: float,
    cam_fov_deg: float,
) -> tuple[LidarAngles, LidarRanges, LidarLabels]:
    """Label LiDAR ranges based on camera labels.

    Args:
        lidar_ranges: LiDAR range measurements
        camera_labels: Labels for camera image (camera_get_labels output)
    Returns:
        Tuple of (lidar_angles, lidar_ranges, lidar_labels):
            lidar_angles: Corresponding LiDAR angles (limited to camera's FoV)
            lidar_ranges: LiDAR ranges (limited to camera's FoV)
            lidar_labels: Labels assigned to each LiDAR point based on the majority label in the corresponding camera segment
    """
    # Lidar angles
    lidar_angles = np.linspace(
        -lidar_fov_deg / 2,
        lidar_fov_deg / 2,
        len(lidar_ranges),
    )

    # Restrict to angles within the camera's field of view
    lidar_in_range_mask = (-cam_fov_deg / 2 <= lidar_angles) & (
        lidar_angles <= cam_fov_deg / 2
    )
    lidar_angles_in_fov = lidar_angles[lidar_in_range_mask]
    lidar_ranges_in_fov = lidar_ranges[lidar_in_range_mask]
    nl = len(lidar_angles_in_fov)

    # Split camera label image into nl segments
    cam_labels_segments = np.array_split(camera_labels, nl, axis=1)

    # Assign LiDAR labels based on the majority label in the corresponding camera segment
    lidar_labels = np.zeros_like(lidar_ranges_in_fov, dtype=np.uint8)
    for i in range(nl):
        segment_labels = cam_labels_segments[i]
        if segment_labels.size == 0:
            continue
        majority_label = np.bincount(segment_labels.flatten()).argmax()
        lidar_labels[i] = majority_label

    ranges_ = cast(LidarRanges, lidar_ranges_in_fov)
    angles_ = cast(LidarAngles, lidar_angles_in_fov)
    labels_ = cast(LidarLabels, lidar_labels)
    return angles_, ranges_, labels_


def clustering(
    points: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    eps: float = 0.5,
    min_samples: int = 3,
) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
    """Cluster points using a spatial-index based DBSCAN variant.

    Args:
        points: Input points to cluster (shape: [n_points, n_features])
        eps: Maximum distance between two samples for them to be considered
            as in the same neighbourhood (same units as ``points``).
        min_samples: Minimum number of samples in a neighbourhood for a
            point to be considered a core point.
    Returns:
        Cluster labels for each point (shape: [n_points], dtype: int32).
        Noise points are labelled ``-1`` as in sklearn.
    """
    # empty input -> empty output
    n_pts = len(points)
    if n_pts == 0:
        return np.empty((0,), dtype=np.int32)

    tree = cKDTree(points)
    # for every point, list of indices within eps
    neighbours = tree.query_ball_point(points, r=eps)

    # simple union-find for grouping reachable core points
    parent = list(range(n_pts))

    def find(i: int) -> int:
        # path compression
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    # mark core points and union neighbouring core points
    is_core = [len(neighbours[i]) >= min_samples for i in range(n_pts)]
    for i in range(n_pts):
        if not is_core[i]:
            continue
        for j in neighbours[i]:
            if is_core[j]:
                union(i, j)

    labels = -np.ones(n_pts, dtype=np.int32)
    cluster_id = 0
    for i in range(n_pts):
        if not is_core[i]:
            # noise will remain -1
            continue
        root = find(i)
        if labels[root] == -1:
            labels[root] = cluster_id
            cluster_id += 1
        labels[i] = labels[root]

    return labels


def get_opponent_positions(
    lidar_positions: LidarCartesianPositions,
    lidar_labels: LidarLabels,
) -> list[tuple[float, float]]:
    """Get opponent positions in Cartesian coordinates (x, y) based on LiDAR data.

    Args:
        lidar_positions: LiDAR positions in Cartesian coordinates (x, y)
        lidar_labels: LiDAR labels
    Returns:
        Array of (x, y) positions for detected opponents.
    """
    # Filter for opponent points
    opponent_mask = lidar_labels == SegmentationLabels.OPPONENT.value
    opponent_positions = lidar_positions[opponent_mask]

    # Cluster opponent points to identify distinct opponents
    cluster_labels = clustering(opponent_positions)

    # Calculate mean position for each cluster to get opponent positions
    opponent_positions_list: list[tuple[float, float]] = []
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue  # Skip noise points
        cluster_points = opponent_positions[cluster_labels == cluster_id]
        mean_position = np.mean(cluster_points, axis=0)
        opponent_positions_list.append(cast(tuple[float, float], tuple(mean_position)))
    return opponent_positions_list


# ====================================================
#  Tests
# ====================================================
if __name__ == "__main__":
    # ====== Parameters ======
    SEGMENTATION_MASKS: dict[int, HSVMaskRanges] = {
        # SegmentationLabels.FREE.value: ...
        SegmentationLabels.WALL_RED.value: HSVMaskRanges(
            h=(162.0, 12.0), s=(200.0, 255.0), v=(15.0, 255.0)
        ),
        SegmentationLabels.WALL_GREEN.value: HSVMaskRanges(
            h=(33.0, 72.0), s=(157.0, 255.0), v=(47.0, 255.0)
        ),
        # SegmentationLabels.FLOOR.value: ...
        # SegmentationLabels.MISC_OBSTACLE.value: ...
    }

    CAM_BAND_Y_MIN = 20
    CAM_BAND_Y_MAX = 25
    CAM_FOV_DEG = np.rad2deg(0.785398)  # Field of view of the camera (degrees)
    LIDAR_FOV_DEG = 360  # Field of view of the LIDAR (degrees), centered on the front of the car, so e.g. 270 means 135 degrees to the left and 135 degrees to the right. Note: this is not necessarily the same as the camera FoV, which may be wider or narrower.

    # ====== Main ======

    from pathlib import Path
    from matplotlib import pyplot as plt

    # Open camera_capture.png
    print("Opening sample camera image...")
    image_path = Path(__file__).parent / "test_resources" / "camera_capture.png"
    image_bgr = cv2.imread(str(image_path))
    assert image_bgr is not None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb_array = cast(ImageArrayRGB, image_rgb)

    # Open lidar_scan.txt
    print("Opening sample LiDAR scan...")
    lidar_path = Path(__file__).parent / "test_resources" / "lidar_scan.txt"
    lidar_data = np.loadtxt(lidar_path, delimiter=",")
    lidar_ranges = cast(LidarRanges, lidar_data)

    # Test camera_get_labels
    print("Testing camera_get_labels...")
    camera_labels = camera_get_labels(
        image_rgb_array, CAM_BAND_Y_MIN, CAM_BAND_Y_MAX, SEGMENTATION_MASKS
    )
    if True:
        # Display only
        camera_labels_image = np.zeros((*camera_labels.shape, 3), dtype=np.uint8)
        camera_labels_image[camera_labels == SegmentationLabels.WALL_RED.value] = [
            255,
            0,
            0,
        ]
        camera_labels_image[camera_labels == SegmentationLabels.WALL_GREEN.value] = [
            0,
            255,
            0,
        ]
        camera_labels_image[camera_labels == SegmentationLabels.OPPONENT.value] = [
            0,
            0,
            255,
        ]
        plt.imshow(camera_labels_image)
        plt.title("Camera Labels")
        plt.axis("off")
        plt.show()

    # Test lidar_get_labelled_ranges
    print("Testing lidar_get_labelled_ranges...")
    lidar_angles, lidar_ranges_in_fov, lidar_labels = lidar_get_labelled_ranges(
        lidar_ranges, camera_labels, LIDAR_FOV_DEG, CAM_FOV_DEG
    )
    if True:
        # Display only
        plt.scatter(lidar_angles, lidar_ranges_in_fov, c=lidar_labels, cmap="tab10")
        plt.title("LiDAR Ranges with Labels")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Range (meters)")
        plt.colorbar(label="Label")
        plt.show()

    # Test get_opponent_positions
    print("Testing get_opponent_positions...")
    # Convert to Cartesian coordinates (x, y)
    lidar_angles_rad = np.deg2rad(lidar_angles)
    lidar_x = lidar_ranges_in_fov * np.sin(lidar_angles_rad)
    lidar_y = lidar_ranges_in_fov * np.cos(lidar_angles_rad)
    lidar_positions = cast(
        LidarCartesianPositions, np.stack((lidar_x, lidar_y), axis=-1)
    )
    opponent_positions = get_opponent_positions(lidar_positions, lidar_labels)
    if True:
        # Display only
        # Show labellet points and opponent positions
        plt.scatter(
            lidar_positions[:, 0],
            lidar_positions[:, 1],
            c=lidar_labels,
            cmap="tab10",
            label="LiDAR Points",
        )
        if opponent_positions:
            opponent_positions_array = np.array(opponent_positions)
            plt.scatter(
                opponent_positions_array[:, 0],
                opponent_positions_array[:, 1],
                c="red",
                marker="x",
                label="Opponent Positions",
            )
        plt.title("LiDAR Ranges with Labels and Opponent Positions")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.legend()
        plt.show()
