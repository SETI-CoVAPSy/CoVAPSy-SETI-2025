"""
Common definitions and utilities for the Super Mega Fusion project.
"""

from enum import IntEnum
from numpy import ndarray, dtype
import numpy as np

# ====================================================
#  Parameters
# ====================================================

# ====================================================
#  Types
# ====================================================

# ===== Base =====
ImageArrayRGB = ndarray[tuple[int, int, int], dtype[np.uint8]]
ImageArrayHSV = ndarray[tuple[int, int, int], dtype[np.float32]]
ImageArrayMask = ndarray[tuple[int, int], dtype[np.bool_]]
LidarRanges = ndarray[tuple[int], dtype[np.float32]]
LidarAngles = ndarray[tuple[int], dtype[np.float32]]
LidarCartesianPositions = ndarray[tuple[int, 2], dtype[np.float32]]


# ===== Labels =====
ImageLabels = ndarray[tuple[int, int], dtype[np.uint8]]  # uint8: SegmentationLabels
LidarLabels = ndarray[tuple[int], dtype[np.uint8]]  # uint8: SegmentationLabels


class SegmentationLabels(IntEnum):
    """Available labels"""

    # SegmentationLabels_Type values
    FREE = 0
    MISC_OBSTACLE = 1
    WALL_RED = 2
    WALL_GREEN = 3
    FLOOR = 4
    OPPONENT = 255


# ====================================================
#  Classes
# ====================================================

# ====================================================
#  Tests
# ====================================================
if __name__ == "__main__":
    pass
