import numpy as np
import cv2
from numpy.typing import NDArray
from enum import Enum
from dataclasses import dataclass
from typing import TypedDict, TypeAlias

class Labels(Enum):
    BACKGROUND = 0
    WALL_RED = 1
    WALL_GREEN = 2
    ENEMY = 3

@dataclass
class ChannelIntervals:
    h: tuple[float, float]
    s: tuple[float, float]
    v: tuple[float, float]

VisionLabelIntervals: TypeAlias = dict[Labels, ChannelIntervals]


def rgb_to_hsv_normalized(rgb_image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Convert an RGB image (uint8) to HSV format with float32 values normalized to [0, 1]."""
    assert rgb_image.dtype == np.uint8, "Input image must be of type uint8."
    assert len(rgb_image.shape) == 3, "Input image must have three dimensions (height, width, channels)."
    assert rgb_image.shape[2] == 3, "Input image must have three channels (RGB)."

    # Convert RGB to BGR for OpenCV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    # Normalize HSV to [0, 1]
    hsv_float = hsv_image.astype(np.float32)
    hsv_float[:, :, 0] /= 179.0  # Hue range in OpenCV is [0, 179]
    hsv_float[:, :, 1] /= 255.0  # Saturation range in OpenCV is [0, 255]
    hsv_float[:, :, 2] /= 255.0  # Value range in OpenCV is [0, 255]

    return hsv_float

def hsv_mask(
        hsv_image: NDArray[np.float32],
        h_interval: tuple[float, float], # Accepts rollong from 1 to 0
        s_interval: tuple[float, float],
        v_interval: tuple[float, float],
    ) -> NDArray[np.bool_]:
    """Get a mask for the given HSV image based on the provided intervals.
    
    Args:
        hsv_image: The input image in HSV color space.
        h_interval (tuple[float, float]): lower and upper bounds for the hue channel (0 to 1).
        s_interval (tuple[float, float]): lower and upper bounds for the saturation channel (0 to 1).
        v_interval (tuple[float, float]): lower and upper bounds for the value channel (0 to 1, can wrap around).
    """
    assert hsv_image.dtype == np.float32, "Input image must be of type float32."
    assert np.all((hsv_image >= 0.0) & (hsv_image <= 1.0)), "Input image values must be in the range [0, 1]."
    assert len(hsv_image.shape) == 3, "Input image must have three dimensions (height, width, channels)."
    assert hsv_image.shape[2] == 3, "Input image must have three channels (HSV)."

    h, s, v = cv2.split(hsv_image)
    h_mask = np.logical_or(
        np.logical_and(h >= h_interval[0], h <= 1.0),
        np.logical_and(h >= 0.0, h <= h_interval[1])
    ) if h_interval[0] > h_interval[1] else np.logical_and(h >= h_interval[0], h <= h_interval[1])

    s_mask = np.logical_and(s >= s_interval[0], s <= s_interval[1])
    v_mask = np.logical_and(v >= v_interval[0], v <= v_interval[1])

    return np.logical_and(h_mask, np.logical_and(s_mask, v_mask))

def image_to_labels(
    rgb_image: NDArray[np.uint8],
    label_intervals: VisionLabelIntervals
) -> NDArray[np.uint8]:
    """Converts the image band to a label array based on predefined HSV intervals for each class.
    
    For now, enemy is what's not marked by other labels."""
    assert rgb_image.dtype == np.uint8, "Input image must be of type uint8."
    assert len(rgb_image.shape) == 3, "Input image must have three dimensions (height, width, channels)."
    assert rgb_image.shape[2] == 3, "Input image must have three channels (BGR)."
    assert all(label in label_intervals for label in Labels), "All labels must have defined intervals."

    # Convert to normalized HSV
    hsv_band_image = rgb_to_hsv_normalized(rgb_image)

    # Get masks
    masks: dict[Labels, NDArray[np.bool_]] = {
        label: hsv_mask(hsv_band_image, intervals.h, intervals.s, intervals.v)
        for label, intervals in label_intervals.items()
    }

    # Create label array (with priority order) 
    label_array = np.zeros(rgb_image.shape[:2], dtype=np.uint8) # Initialize a (height, width) array for labels
    label_array[:] = Labels.ENEMY.value # default to enemy
    label_array[masks[Labels.WALL_GREEN]] = Labels.WALL_GREEN.value
    label_array[masks[Labels.WALL_RED]] = Labels.WALL_RED.value
    label_array[masks[Labels.BACKGROUND]] = Labels.BACKGROUND.value # Force background too
    return label_array

# def flatten_labels(label_array: NDArray[np.uint8]) -> NDArray[np.uint8]:
#     """Flattens the label array to 1D."""
#     assert label_array.dtype == np.uint8, "Input label array must be of type uint8."
#     assert len(label_array.shape) == 2, "Input label array must have two dimensions (height, width)."

#     # Flatten to 1D (decision step): max of each column
#     label_array_1d = label_array.max(axis=0)

#     return label_array_1d

def fusion_label(
    rgb_image: NDArray[np.uint8],
    image_fov_angle: float, # theta
    array_lidar: NDArray[np.float32], # 1D array of ranges 
    lidar_min_angle: float,
    lidar_max_angle: float,
    label_intervals: VisionLabelIntervals,
    cam_plane_distance: float = 0.05 # in m, distance from camera position to the focal plane
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]]: # labels, angles, ranges
    """Performs our data fusion algorithm."""
    assert rgb_image.dtype == np.uint8, "Input image must be of type uint8."
    assert len(rgb_image.shape) == 3, "Input image must have three dimensions (height, width, channels)."
    assert rgb_image.shape[2] == 3, "Input image must have three channels (BGR)."
    assert all(label in label_intervals for label in Labels), "All labels must have defined intervals."
    assert 0 < image_fov_angle < lidar_max_angle - lidar_min_angle <= 360, "Field of view angles must be in the range (0, 360] and image FOV must be less than lidar FOV." 

    # Extract a band 
    band_y_min = int(rgb_image.shape[0] * 0.45)
    band_y_max = int(rgb_image.shape[0] * 0.55)
    image_band = rgb_image[band_y_min:band_y_max, :, :]

    # Get lidar ranges within the image FOV
    angle_per_index = (lidar_max_angle - lidar_min_angle) / len(array_lidar)
    lidar_angles = np.arange(lidar_min_angle, lidar_max_angle, angle_per_index)
    lidar_in_fov_mask = np.logical_and(lidar_angles >= -image_fov_angle / 2, lidar_angles <= image_fov_angle / 2)
    lidar_ranges_in_fov = array_lidar[lidar_in_fov_mask]
    lidar_angles_in_fov = lidar_angles[lidar_in_fov_mask]

    # Get labels for the band
    label_array_band = image_to_labels(image_band, label_intervals)

    def lidar_angle_to_pixel_fraction(angle: float) -> float:
        """Get pixel coordinate corresponding to a lidar angle, as a fraction of the image width centered on the middle of the image."""
        return cam_plane_distance * np.tan(np.radians(angle) / 2)
    
    # Cut the label band into vertical strips according to lidar angles
    # Get middle of strips
    strip_pixel_indices = []
    for i, angle in enumerate(lidar_angles_in_fov):
        pixel_fraction = lidar_angle_to_pixel_fraction(angle)
        pixel_index = int((pixel_fraction + 0.5) * label_array_band.shape[1]) # Convert from fraction to pixel index
        strip_pixel_indices.append(pixel_index)
    strip_pixel_indices = [0] + strip_pixel_indices + [label_array_band.shape[1]] # Add start and end of image as strip boundaries
    
    # Check matching strip and lidar angle count
    assert len(strip_pixel_indices) - 1 == len(lidar_angles_in_fov), "Number of strips must match number of lidar angles in FOV."

    # For each strip, get the most common label and assign it to the corresponding lidar angle
    label_array_assigned_to_lidar_angles = np.zeros_like(lidar_angles_in_fov, dtype=np.uint8) # Initialize an array to hold the assigned labels for each lidar angle
    for i in range(len(strip_pixel_indices) - 1):
        strip_start = strip_pixel_indices[i]
        strip_end = strip_pixel_indices[i + 1]
        strip_labels = label_array_band[:, strip_start:strip_end]
        most_common_label = np.bincount(strip_labels.flatten()).argmax() # Get the most common label in the strip
        label_array_assigned_to_lidar_angles[i] = most_common_label
    
    return label_array_assigned_to_lidar_angles, lidar_angles_in_fov, lidar_ranges_in_fov
