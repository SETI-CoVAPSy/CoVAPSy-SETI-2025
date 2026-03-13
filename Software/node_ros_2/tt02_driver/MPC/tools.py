"""Shared helpers for robust ROS parameter parsing and lidar transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", "", "none", "null"}


@dataclass(frozen=True)
class LidarCalibration:
    """Normalized lidar scan calibration parameters."""

    angle_offset_rad: float
    mirror: bool
    reverse: bool
    offset_x: float
    offset_y: float


def to_bool(value: object, default: bool = False) -> bool:
    """Convert ROS parameter-like values to bool safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_STRINGS:
            return True
        if normalized in FALSE_STRINGS:
            return False
    return default


def get_bool_param(node: object, name: str, default: bool = False) -> bool:
    """Read a ROS parameter and normalize it to bool."""
    return to_bool(getattr(node.get_parameter(name), "value", None), default)


def get_float_param(node: object, name: str, default: float = 0.0) -> float:
    """Read a ROS parameter and normalize it to float."""
    value = getattr(node.get_parameter(name), "value", None)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_lidar_calibration(
    node: object,
    *,
    default_mirror: bool = True,
    default_reverse: bool = False,
) -> LidarCalibration:
    """Read lidar calibration parameters from ROS each time they are needed."""
    return LidarCalibration(
        angle_offset_rad=math.radians(get_float_param(node, "scan_angle_offset_deg", 0.0)),
        mirror=get_bool_param(node, "scan_mirror", default_mirror),
        reverse=get_bool_param(node, "scan_reverse", default_reverse),
        offset_x=get_float_param(node, "lidar_offset_x", 0.0),
        offset_y=get_float_param(node, "lidar_offset_y", 0.0),
    )


def laser_scan_to_vehicle_frame(
    scan_msg: object,
    calibration: LidarCalibration,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a LaserScan into the vehicle frame using shared calibration."""
    if not getattr(scan_msg, "ranges", None):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    ranges = np.asarray(scan_msg.ranges, dtype=float)
    angles = scan_msg.angle_min + np.arange(ranges.size, dtype=float) * scan_msg.angle_increment

    if calibration.reverse:
        ranges = ranges[::-1]
        angles = angles[::-1]

    angles = angles + calibration.angle_offset_rad

    stride = max(1, int(stride))
    if stride > 1:
        idx = np.arange(0, ranges.size, stride, dtype=int)
        ranges = ranges[idx]
        angles = angles[idx]

    valid = (
        np.isfinite(ranges)
        & (ranges >= scan_msg.range_min)
        & (ranges <= scan_msg.range_max)
    )
    if not np.any(valid):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    ranges = ranges[valid]
    angles = angles[valid]

    x_local = ranges * np.cos(angles)
    y_local = ranges * np.sin(angles)

    if calibration.mirror:
        y_local = -y_local

    x_vehicle = calibration.offset_x + x_local
    y_vehicle = calibration.offset_y + y_local
    return x_vehicle, y_vehicle


def laser_scan_to_world_frame(
    scan_msg: object,
    pose: tuple[float, float, float],
    calibration: LidarCalibration,
    *,
    stride: int = 1,
) -> np.ndarray:
    """Project a LaserScan into world coordinates using shared calibration."""
    x_vehicle, y_vehicle = laser_scan_to_vehicle_frame(
        scan_msg,
        calibration,
        stride=stride,
    )
    if x_vehicle.size == 0:
        return np.empty((0, 2), dtype=float)

    x_car, y_car, yaw = pose
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x_world = x_car + cos_y * x_vehicle - sin_y * y_vehicle
    y_world = y_car + sin_y * x_vehicle + cos_y * y_vehicle
    return np.column_stack((x_world, y_world))
