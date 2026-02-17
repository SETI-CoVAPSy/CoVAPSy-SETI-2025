"""
Generate a Webots world.
"""

import numpy as np

from pathlib import Path
from typing import Literal
from PIL import Image

from lib_track_generator import TrackGenerator, VALUE_TO_DIRECTION, DIRECTION_TO_VALUE
from lib_webots_world import (
    World,
    CoVAPSyCar,
    Floor,
    BorderStraight,
    BorderCurve,
    ComponentBase,
    ComponentGroup,
)

# =============================================
#  User parameters
# =============================================
MAP_WIDTH = 3  # in track units
MAP_HEIGHT = 5  # in track units
TRACK_MARGIN = min(MAP_WIDTH, MAP_HEIGHT) // 5  # in track units
TRACK_SCALE = 2.0  # in meters, should be above 2.0
TRACK_GENERATION_STEPS = (MAP_WIDTH * MAP_HEIGHT) * 5

PATH_WORLD = Path(__file__).parent / "Webots_SETI_gen" / "worlds"
PATH_WBT = PATH_WORLD / "CoVAPSy_SETI_2025_generated.wbt"
PATH_TRACK_IMAGE = PATH_WORLD / "generated_track_preview.png"
PATH_TEST_TRACK_IMAGE = (
    Path(__file__).parent / "track_template_build_test.png"
)  # Track template to import from

DO_TEST_TRACK = False  # If True, load track from test image instead of generating it
DO_FOLLOW = False # If True, viewpoint follow

class GridTrackStraight(ComponentBase):
    """Straight track component."""

    def __init__(
        self,
        name: str,
        direction: Literal["N", "E", "S", "W"],
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        length: int = 1,
        # length: float = TRACK_SCALE
    ) -> None:
        self.name = name
        track_red: BorderStraight
        track_green: BorderStraight
        if direction == "N":
            track_red = BorderStraight(
                f"{name}_Red",
                length=TRACK_SCALE*length,
                translation=(
                    translation[0] - 0.25 * TRACK_SCALE,
                    translation[1] + TRACK_SCALE * (length - 1.0)/2,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(1.0, 0.0, 0.0),
            )
            track_green = BorderStraight(
                f"{name}_Green",
                length=TRACK_SCALE*length,
                translation=(
                    translation[0] + 0.25 * TRACK_SCALE,
                    translation[1] + TRACK_SCALE * (length - 1.0)/2,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(0.0, 1.0, 0.0),
            )
        elif direction == "E":
            track_red = BorderStraight(
                f"{name}_Red",
                length=TRACK_SCALE*length,
                translation=(
                    translation[0] + TRACK_SCALE * (length - 1.0)/2,
                    translation[1] + 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0),
            )
            track_green = BorderStraight(
                f"{name}_Green",
                length=TRACK_SCALE*length,
                translation=(
                    translation[0] + TRACK_SCALE * (length - 1.0)/2,
                    translation[1] - 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0),
            )
        elif direction == "S":
            track_red = BorderStraight(
                f"{name}_Red",
                length=TRACK_SCALE*length,
                translation=(
                    translation[0] + 0.25 * TRACK_SCALE,
                    translation[1] - TRACK_SCALE * (length - 1.0)/2,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(1.0, 0.0, 0.0),
            )
            track_green = BorderStraight(
                f"{name}_Green",
                length=TRACK_SCALE*length,
                translation=(
                    translation[0] - 0.25 * TRACK_SCALE,
                    translation[1] - TRACK_SCALE * (length - 1.0)/2,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(0.0, 1.0, 0.0),
            )
        elif direction == "W":
            track_red = BorderStraight(
                f"{name}_Red",
                length=TRACK_SCALE*length,
                translation=(
                    translation[0] - TRACK_SCALE * (length - 1.0)/2,
                    translation[1] - 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0),
            )
            track_green = BorderStraight(
                f"{name}_Green",
                length=TRACK_SCALE*length,
                translation=(
                    translation[0] - TRACK_SCALE * (length - 1.0)/2,
                    translation[1] + 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0),
            )
        else:
            raise ValueError(f"Invalid direction: {direction}")

        self.track_components = ComponentGroup([track_red, track_green], name=self.name)

    def get_string(self, indent: str = "  ") -> str:
        """Get string representation."""
        return self.track_components.get_string(indent=indent)
class GridTrackCurve(ComponentBase):
    """Curved track component."""

    EPSILON = 1e-2

    def __init__(
        self,
        name: str,
        direction_in: Literal["N", "E", "S", "W"],
        direction_out: Literal["N", "E", "S", "W"],
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.name = name



        curve_red: BorderCurve
        curve_green: BorderCurve
        straight_red1: BorderStraight
        straight_green1: BorderStraight
        straight_red2: BorderStraight
        straight_green2: BorderStraight
        if direction_in == "N" and direction_out == "E":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0]-(TRACK_SCALE-2.0)/4, translation[1]+(TRACK_SCALE-2.0)/4, translation[2] + 0.1),
                rotation=(0.0, 0.0, 1.0, 3 * np.pi / 2 ),
                color=(1.0, 0.0, 0.0),
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(
                    translation[0] + 0.5 * TRACK_SCALE-(TRACK_SCALE-2.0)/4,
                    translation[1] - 0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                    translation[2] + 0.1,
                ),
                rotation=(0.0, 0.0, 1.0, 3 * np.pi / 2),
                color=(0.0, 1.0, 0.0),
            )
            straight_red1 = BorderStraight(
                f"{name}_Red_Straight1",
                length=0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] - 0.25 * TRACK_SCALE,
                    translation[1] - 0.25 * TRACK_SCALE+(TRACK_SCALE-2.0)/8,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(1.0, 0.0, 0.0),
            )
            straight_red2 = BorderStraight(
                f"{name}_Red_Straight2",
                length=0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] + 0.25 * TRACK_SCALE-(TRACK_SCALE-2.0)/8,
                    translation[1] + 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0),
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_red1, straight_red2], name=self.name
            )
            if TRACK_SCALE - 2.0 > self.EPSILON:
                straight_green1 = BorderStraight(
                    f"{name}_Green_Straight1",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] + 0.25 * TRACK_SCALE,
                        translation[1] - 0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/8,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, np.pi / 2),
                    color=(0.0, 1.0, 0.0),
                )
                straight_green2 = BorderStraight(
                    f"{name}_Green_Straight2",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] + 0.5 * TRACK_SCALE-(TRACK_SCALE-2.0)/8,
                        translation[1] - 0.25 * TRACK_SCALE,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, 0.0),
                    color=(0.0, 1.0, 0.0),
                )
                self.track_components.children.extend([straight_green1, straight_green2])
        elif direction_in == "S" and direction_out == "E":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(
                    translation[0] + 0.5 * TRACK_SCALE-(TRACK_SCALE-2.0)/4,
                    translation[1] + 0.5 * TRACK_SCALE-(TRACK_SCALE-2.0)/4,
                    translation[2] + 0.1,
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0),
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0]-(TRACK_SCALE-2.0)/4, translation[1]-(TRACK_SCALE-2.0)/4, translation[2] + 0.1),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0),
            )
            straight_green1 = BorderStraight(
                f"{name}_Green_Straight1",
                length=0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] - 0.25 * TRACK_SCALE,
                    translation[1] + 0.25 * TRACK_SCALE-(TRACK_SCALE-2.0)/8,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(0.0, 1.0, 0.0),
            )
            straight_green2 = BorderStraight(
                f"{name}_Green_Straight2",
                length=0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] + 0.25 * TRACK_SCALE-(TRACK_SCALE-2.0)/8,
                    translation[1] - 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0),
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_green1, straight_green2], name=self.name
            )
            if TRACK_SCALE - 2.0 > self.EPSILON:
                straight_red1 = BorderStraight(
                    f"{name}_Red_Straight1",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] + 0.25 * TRACK_SCALE,
                        translation[1] + 0.5 * TRACK_SCALE-(TRACK_SCALE-2.0)/8,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, np.pi / 2),
                    color=(1.0, 0.0, 0.0),
                )
                straight_red2 = BorderStraight(
                    f"{name}_Red_Straight2",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] + 0.5 * TRACK_SCALE-(TRACK_SCALE-2.0)/8,
                        translation[1] + 0.25 * TRACK_SCALE,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, 0.0),
                    color=(1.0, 0.0, 0.0),
                )
                self.track_components.children.extend([straight_red1, straight_red2])
        elif direction_in == "E" and direction_out == "S":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0]+(TRACK_SCALE-2.0)/4, translation[1]+(TRACK_SCALE-2.0)/4, translation[2] + 0.1),
                rotation=(0.0, 0.0, 1.0, np.pi),
                color=(1.0, 0.0, 0.0),
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(
                    translation[0] - 0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                    translation[1] - 0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                    translation[2] + 0.1,
                ),
                rotation=(0.0, 0.0, 1.0, np.pi),
                color=(0.0, 1.0, 0.0),
            )
            straight_red1 = BorderStraight(
                f"{name}_Red_Straight1",
                length=0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] + 0.25 * TRACK_SCALE,
                    translation[1] - 0.25 * TRACK_SCALE+(TRACK_SCALE-2.0)/8,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(1.0, 0.0, 0.0),
            )
            straight_red2 = BorderStraight(
                f"{name}_Red_Straight2",
                length=0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] - 0.25 * TRACK_SCALE+(TRACK_SCALE-2.0)/8,
                    translation[1] + 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0),
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_red1, straight_red2], name=self.name
            )
            if TRACK_SCALE - 2.0 > self.EPSILON:
                straight_green1 = BorderStraight(
                    f"{name}_Green_Straight1",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] - 0.25 * TRACK_SCALE,
                        translation[1] - 0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/8,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, np.pi / 2),
                    color=(0.0, 1.0, 0.0),
                )
                straight_green2 = BorderStraight(
                    f"{name}_Green_Straight2",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] - 0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/8,
                        translation[1] - 0.25 * TRACK_SCALE,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, 0.0),
                    color=(0.0, 1.0, 0.0),
                )
                self.track_components.children.extend([straight_green1, straight_green2])
        elif direction_in == "W" and direction_out == "S":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(
                    translation[0] + 0.5 * TRACK_SCALE-(TRACK_SCALE-2.0)/4,
                    translation[1] - 0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                    translation[2] + 0.1,
                ),
                rotation=(0.0, 0.0, 1.0, 3 * np.pi / 2),
                color=(1.0, 0.0, 0.0),
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0]-(TRACK_SCALE-2.0)/4, translation[1]+(TRACK_SCALE-2.0)/4, translation[2] + 0.1),
                rotation=(0.0, 0.0, 1.0, 3 * np.pi / 2),
                color=(0.0, 1.0, 0.0),
            )
            straight_green1 = BorderStraight(
                f"{name}_Green_Straight1",
                length=0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] - 0.25 * TRACK_SCALE,
                    translation[1] - 0.25 * TRACK_SCALE+(TRACK_SCALE-2.0)/8,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(0.0, 1.0, 0.0),
            )
            straight_green2 = BorderStraight(
                f"{name}_Green_Straight2",
                length=0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] + 0.25 * TRACK_SCALE-(TRACK_SCALE-2.0)/8,
                    translation[1] + 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0),
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_green1, straight_green2], name=self.name
            )
            if TRACK_SCALE - 2.0 > self.EPSILON:
                straight_red1 = BorderStraight(
                    f"{name}_Red_Straight1",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] + 0.25 * TRACK_SCALE,
                        translation[1] - 0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/8,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, np.pi / 2),
                    color=(1.0, 0.0, 0.0),
                )
                straight_red2 = BorderStraight(
                    f"{name}_Red_Straight2",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] + 0.5 * TRACK_SCALE-(TRACK_SCALE-2.0)/8,
                        translation[1] - 0.25 * TRACK_SCALE,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, 0.0),
                    color=(1.0, 0.0, 0.0),
                )
                self.track_components.children.extend([straight_red1, straight_red2])
        elif direction_in == "S" and direction_out == "W":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0]+(TRACK_SCALE-2.0)/4, translation[1]-(TRACK_SCALE-2.0)/4, translation[2] + 0.1),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(1.0, 0.0, 0.0),
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(
                    translation[0] - 0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                    translation[1] + 0.5 * TRACK_SCALE-(TRACK_SCALE-2.0)/4,
                    translation[2] + 0.1,
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(0.0, 1.0, 0.0),
            )
            straight_red1 = BorderStraight(
                f"{name}_Red_Straight1",
                length=0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] + 0.25 * TRACK_SCALE,
                    translation[1] + 0.25 * TRACK_SCALE - (TRACK_SCALE-2.0)/8,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(1.0, 0.0, 0.0),
            )
            straight_red2 = BorderStraight(
                f"{name}_Red_Straight2",
                length=0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] - 0.25 * TRACK_SCALE + (TRACK_SCALE-2.0)/8,
                    translation[1] - 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0),
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_red1, straight_red2], name=self.name
            )
            if TRACK_SCALE - 2.0 > self.EPSILON:
                straight_green1 = BorderStraight(
                    f"{name}_Green_Straight1",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] - 0.25 * TRACK_SCALE,
                        translation[1] + 0.5 * TRACK_SCALE - (TRACK_SCALE-2.0)/8,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, np.pi / 2),
                    color=(0.0, 1.0, 0.0),
                )
                straight_green2 = BorderStraight(
                    f"{name}_Green_Straight2",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] - 0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/8,
                        translation[1] + 0.25 * TRACK_SCALE,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, 0.0),
                    color=(0.0, 1.0, 0.0),
                )
                self.track_components.children.extend([straight_green1, straight_green2])     
        elif direction_in == "N" and direction_out == "W":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(
                    translation[0] - 0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/4,
                    translation[1] - 0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/4,
                    translation[2] + 0.1,
                ),
                rotation=(0.0, 0.0, 1.0, np.pi),
                color=(1.0, 0.0, 0.0),
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0] + (TRACK_SCALE-2.0)/4, translation[1] + (TRACK_SCALE-2.0)/4, translation[2] + 0.1),
                rotation=(0.0, 0.0, 1.0, np.pi),
                color=(0.0, 1.0, 0.0),
            )
            straight_green1 = BorderStraight(
                f"{name}_Green_Straight1",
                length=0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] + 0.25 * TRACK_SCALE,
                    translation[1] - 0.25 * TRACK_SCALE + (TRACK_SCALE-2.0)/8,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(0.0, 1.0, 0.0),
            )
            straight_green2 = BorderStraight(
                f"{name}_Green_Straight2",
                length=0.5 * TRACK_SCALE+(TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] - 0.25 * TRACK_SCALE + (TRACK_SCALE-2.0)/8,
                    translation[1] + 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0),
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_green1, straight_green2], name=self.name
            )
            if TRACK_SCALE - 2.0 > self.EPSILON:
                straight_red1 = BorderStraight(
                    f"{name}_Red_Straight1",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] - 0.25 * TRACK_SCALE,
                        translation[1] - 0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/8,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, np.pi / 2),
                    color=(1.0, 0.0, 0.0),
                )
                straight_red2 = BorderStraight(
                    f"{name}_Red_Straight2",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] - 0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/8,
                        translation[1] - 0.25 * TRACK_SCALE,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, 0.0),
                    color=(1.0, 0.0, 0.0),
                )
                self.track_components.children.extend([straight_red1, straight_red2])
        elif direction_in == "E" and direction_out == "N":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(
                    translation[0] - 0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/4,
                    translation[1] + 0.5 * TRACK_SCALE - (TRACK_SCALE-2.0)/4,
                    translation[2] + 0.1,
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(1.0, 0.0, 0.0),
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0] + (TRACK_SCALE-2.0)/4, translation[1]-(TRACK_SCALE-2.0)/4, translation[2] + 0.1),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(0.0, 1.0, 0.0),
            )
            straight_green1 = BorderStraight(
                f"{name}_Green_Straight1",
                length=0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] + 0.25 * TRACK_SCALE,
                    translation[1] + 0.25 * TRACK_SCALE - (TRACK_SCALE-2.0)/8,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(0.0, 1.0, 0.0),
            )
            straight_green2 = BorderStraight(
                f"{name}_Green_Straight2",
                length=0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] - 0.25 * TRACK_SCALE + (TRACK_SCALE-2.0)/8,
                    translation[1] - 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0),
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_green1, straight_green2], name=self.name
            )
            if TRACK_SCALE - 2.0 > self.EPSILON:
                straight_red1 = BorderStraight(
                    f"{name}_Red_Straight1",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] - 0.25 * TRACK_SCALE,
                        translation[1] + 0.5 * TRACK_SCALE - (TRACK_SCALE-2.0)/8,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, np.pi / 2),
                    color=(1.0, 0.0, 0.0),
                )
                straight_red2 = BorderStraight(
                    f"{name}_Red_Straight2",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] - 0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/8,
                        translation[1] + 0.25 * TRACK_SCALE,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, 0.0),
                    color=(1.0, 0.0, 0.0),
                )
                self.track_components.children.extend([straight_red1, straight_red2])
        elif direction_in == "W" and direction_out == "N":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0] - (TRACK_SCALE-2.0)/4, translation[1]-(TRACK_SCALE-2.0)/4, translation[2] + 0.1),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0),
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(
                    translation[0] + 0.5 * TRACK_SCALE - (TRACK_SCALE-2.0)/4,
                    translation[1] + 0.5 * TRACK_SCALE - (TRACK_SCALE-2.0)/4,
                    translation[2] + 0.1,
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0),
            )
            straight_red1 = BorderStraight(
                f"{name}_Red_Straight1",
                length=0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] - 0.25 * TRACK_SCALE,
                    translation[1] + 0.25 * TRACK_SCALE - (TRACK_SCALE-2.0)/8,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, np.pi / 2),
                color=(1.0, 0.0, 0.0),
            )
            straight_red2 = BorderStraight(
                f"{name}_Red_Straight2",
                length=0.5 * TRACK_SCALE + (TRACK_SCALE-2.0)/4,
                translation=(
                    translation[0] + 0.25 * TRACK_SCALE - (TRACK_SCALE-2.0)/8,
                    translation[1] - 0.25 * TRACK_SCALE,
                    translation[2],
                ),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0),
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_red1, straight_red2], name=self.name
            )
            if TRACK_SCALE - 2.0 > self.EPSILON:
                straight_green1 = BorderStraight(
                    f"{name}_Green_Straight1",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] + 0.25 * TRACK_SCALE,
                        translation[1] + 0.5 * TRACK_SCALE - (TRACK_SCALE-2.0)/8,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, np.pi / 2),
                    color=(0.0, 1.0, 0.0),
                )
                straight_green2 = BorderStraight(
                    f"{name}_Green_Straight2",
                    length=(TRACK_SCALE-2.0)/4,
                    translation=(
                        translation[0] + 0.5 * TRACK_SCALE - (TRACK_SCALE-2.0)/8,
                        translation[1] + 0.25 * TRACK_SCALE,
                        translation[2],
                    ),
                    rotation=(0.0, 0.0, 1.0, 0.0),
                    color=(0.0, 1.0, 0.0),
                )
                self.track_components.children.extend([straight_green1, straight_green2])
        else:
            self.track_components = ComponentGroup([], name=self.name)

    def get_string(self, indent: str = "  ") -> str:
        """Get string representation."""
        return self.track_components.get_string(indent=indent)

class GridFinishLine(ComponentBase):
    """Finish line component."""

    def __init__(
        self,
        name: str,
        direction: Literal["N", "E", "S", "W"],
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.name = name
        self.direction = direction
        self.translation = translation
        self.color = color

    def get_string(self, indent: str = "  ") -> str:
        """Get string representation."""
        rotation_param = {
            "E": f"0 0 1 {np.pi/2}",
            "S": f"0 0 1 {np.pi}",
            "W": f"0 0 1 {3*np.pi/2}",
            "N": f"0 0 1 0",
        }[self.direction]
        translation_param = {
            "E": f"{self.translation[0]+0.5*TRACK_SCALE} {self.translation[1]} -0.05",
            "S": f"{self.translation[0]} {self.translation[1]-0.5*TRACK_SCALE} -0.05",
            "W": f"{self.translation[0]-0.5*TRACK_SCALE} {self.translation[1]} -0.05",
            "N": f"{self.translation[0]} {self.translation[1]+0.5*TRACK_SCALE} -0.05",
        }[self.direction]

        content = "Solid {\n"
        content += f"{indent}translation {translation_param}\n"
        content += f"{indent}rotation {rotation_param}\n"
        content += f"{indent}children " + "[\n"
        content += f"{indent*2}Shape " + "{\n"
        content += f"{indent*3}appearance PBRAppearance " + "{\n"
        content += (
            f"{indent*4}baseColor {self.color[0]} {self.color[1]} {self.color[2]}\n"
        )
        content += f"{indent*4}metalness 0\n"
        content += f"{indent*3}" + "}\n"
        content += f"{indent*3}geometry Box " + "{\n"
        content += f"{indent*4}size {TRACK_SCALE/2} 0.1 0.01\n"
        content += f"{indent*3}" + "}\n"
        content += f"{indent*2}" + "}\n"
        content += f"{indent}]" + "\n"
        content += f'{indent}name "{self.name}"' + "\n"
        content += "}" + "\n"
        return content


if __name__ == "__main__":
    if not PATH_WORLD.exists():
        raise FileNotFoundError(f"World parent path not found: {PATH_WORLD}")

    # ==== World definition ====
    world: World
    if DO_FOLLOW:
        world = World(world_path=str(PATH_WBT), viewpoint_arg='''
orientation 0.03640700220421551 0.9991312226414955 0.02028127543850064 0.8419091720971831
position -8.000572994132117 5.9031970370616955 4.907303571483722
follow "CoVAPSy_Car_1"
followType "Mounted Shot"''')

    else:
        world = World(world_path=str(PATH_WBT))

    # ==== Track ====
    if not DO_TEST_TRACK:
        # === Generate track ===
        track_generator = TrackGenerator(
            width=MAP_WIDTH, height=MAP_HEIGHT, margin=TRACK_MARGIN
        )
        for _ in range(TRACK_GENERATION_STEPS):
            track_generator.track_gen_step()

    else:  # === Load test track ===
        track_image = Image.open(PATH_TEST_TRACK_IMAGE)
        track_image_data = np.array(track_image)
        # Remove alpha channel if present
        if track_image_data.shape[2] == 4:
            track_image_data = track_image_data[:, :, :3]
        track_generator = TrackGenerator.load_from_matrix_image(track_image_data)

    # ==== Save track image ====
    TrackGenerator.save_track_bitmap_image(
        track=track_generator.track, file_path=str(PATH_TRACK_IMAGE)
    )

    # ==== Populate world ====
    # === Cars ===
    # Get coordinates of starting positions
    east_positions = np.argwhere(track_generator.track == DIRECTION_TO_VALUE["E"])
    start_pos = east_positions[0]  # Defaults to one facing east
    # Try to find three consecutive east-facing positions
    height, width = track_generator.track.shape
    for pos in east_positions:
        y, x = pos
        if (
            0 < x < track_generator.track.shape[1] - 1
            and track_generator.track[y, x - 1] == DIRECTION_TO_VALUE["E"]
            and track_generator.track[y, x + 1] == DIRECTION_TO_VALUE["E"]
        ):
            start_pos = pos
            break
    print("Using starting position: ", start_pos)

    car1 = CoVAPSyCar(
        name="CoVAPSy_Car_1",
        controller="<extern>",
        color=(1.0, 1.0, 0.0),  # Yellow
        translation=(
            (start_pos[1] - width / 2) * TRACK_SCALE + 0.6,
            (-start_pos[0] + height / 2) * TRACK_SCALE,
            0,
        ),
        # translation=(0.0, -1.0, 0),
    )
    world.components.append(car1)

    car2 = CoVAPSyCar(
        name="Enemy_Car_1",
        controller="<none>",
        color=(0.0, 0.0, 1.0),  # Blue
        translation=(
            (start_pos[1] - width / 2) * TRACK_SCALE,
            (-start_pos[0] + height / 2) * TRACK_SCALE - 0.3,
            0,
        ),
    )
    world.components.append(car2)

    car3 = CoVAPSyCar(
        name="Enemy_Car_2",
        controller="<none>",
        color=(1.0, 0.0, 1.0),  # Magenta
        translation=(
            (start_pos[1] - width / 2) * TRACK_SCALE,
            (-start_pos[0] + height / 2) * TRACK_SCALE + 0.3,
            0,
        ),
    )
    world.components.append(car3)

    # Finish line
    finish_line = GridFinishLine(
        name="Finish_Line",
        direction="E",
        translation=(
            (start_pos[1] - width / 2) * TRACK_SCALE,
            (-start_pos[0] + height / 2) * TRACK_SCALE,
            0,
        ),
        color=(1.0, 1.0, 1.0),
    )
    world.components.append(finish_line)

    # === Floor ===
    floor_size = (float(max(MAP_WIDTH, MAP_HEIGHT)) + 1) * TRACK_SCALE
    floor = Floor(name="Ground_Floor", size=(floor_size, floor_size))
    world.components.append(floor)

    # === Track components ===
    group_track = ComponentGroup(name="Track")
    world.components.append(group_track)
    height, width = track_generator.track.shape

    # Track positions
    track_positions_raw = np.argwhere(track_generator.track != DIRECTION_TO_VALUE[None])
    track_positions_set: set[tuple[int, int]] = set(
        (y, x) for y, x in track_positions_raw
    )

    # First, add curves
    for y in range(height):
        for x in range(width):
            cell_value = track_generator.track[y, x]
            direction = VALUE_TO_DIRECTION[cell_value]
            if direction is None:
                continue  # No track

            # Curves
            if (  # N->E
                direction == "E"
                and y < height - 1
                and track_generator.track[y + 1, x] == DIRECTION_TO_VALUE["N"]
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in="N",
                    direction_out="E",
                    translation=(
                        (x - width / 2) * TRACK_SCALE,
                        (-y + height / 2) * TRACK_SCALE,
                        0.0,
                    ),
                )
                group_track.children.append(track_component)
                track_positions_set.discard((y, x))
                continue
            elif (  # S -> E
                direction == "E"
                and y > 0
                and track_generator.track[y - 1, x] == DIRECTION_TO_VALUE["S"]
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in="S",
                    direction_out="E",
                    translation=(
                        (x - width / 2) * TRACK_SCALE,
                        (-y + height / 2) * TRACK_SCALE,
                        0.0,
                    ),
                )
                group_track.children.append(track_component)
                track_positions_set.discard((y, x))
                continue
            elif (  # E -> S
                direction == "S"
                and x > 0
                and track_generator.track[y, x - 1] == DIRECTION_TO_VALUE["E"]
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in="E",
                    direction_out="S",
                    translation=(
                        (x - width / 2) * TRACK_SCALE,
                        (-y + height / 2) * TRACK_SCALE,
                        0.0,
                    ),
                )
                group_track.children.append(track_component)
                track_positions_set.discard((y, x))
                continue
            elif (  # W -> S
                direction == "S"
                and x < width - 1
                and track_generator.track[y, x + 1] == DIRECTION_TO_VALUE["W"]
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in="W",
                    direction_out="S",
                    translation=(
                        (x - width / 2) * TRACK_SCALE,
                        (-y + height / 2) * TRACK_SCALE,
                        0.0,
                    ),
                )
                group_track.children.append(track_component)
                track_positions_set.discard((y, x))
                continue
            elif (  # S -> W
                direction == "W"
                and y > 0
                and track_generator.track[y - 1, x] == DIRECTION_TO_VALUE["S"]
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in="S",
                    direction_out="W",
                    translation=(
                        (x - width / 2) * TRACK_SCALE,
                        (-y + height / 2) * TRACK_SCALE,
                        0.0,
                    ),
                )
                group_track.children.append(track_component)
                track_positions_set.discard((y, x))
                continue
            elif (  # N -> W
                direction == "W"
                and y < height - 1
                and track_generator.track[y + 1, x] == DIRECTION_TO_VALUE["N"]
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in="N",
                    direction_out="W",
                    translation=(
                        (x - width / 2) * TRACK_SCALE,
                        (-y + height / 2) * TRACK_SCALE,
                        0.0,
                    ),
                )
                group_track.children.append(track_component)
                track_positions_set.discard((y, x))
                continue
            elif (  # E -> N
                direction == "N"
                and x > 0
                and track_generator.track[y, x - 1] == DIRECTION_TO_VALUE["E"]
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in="E",
                    direction_out="N",
                    translation=(
                        (x - width / 2) * TRACK_SCALE,
                        (-y + height / 2) * TRACK_SCALE,
                        0.0,
                    ),
                )
                group_track.children.append(track_component)
                track_positions_set.discard((y, x))
                continue
            elif (  # W -> N
                direction == "N"
                and x < width - 1
                and track_generator.track[y, x + 1] == DIRECTION_TO_VALUE["W"]
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in="W",
                    direction_out="N",
                    translation=(
                        (x - width / 2) * TRACK_SCALE,
                        (-y + height / 2) * TRACK_SCALE,
                        0.0,
                    ),
                )
                group_track.children.append(track_component)
                track_positions_set.discard((y, x))
                continue

    # Then, add straights
    # Positions by orientation
    positions_by_orientation: dict[str, set[tuple[int, int]]] = {
        "N": set(),
        "E": set(),
        "S": set(),
        "W": set(),
    }
    for y, x in track_positions_set:
        cell_value = track_generator.track[y, x]
        direction = VALUE_TO_DIRECTION[cell_value]
        if direction is not None:
            positions_by_orientation[direction].add((y, x))
        
    # Now, process all remaining positions
    # East: sort by increasing x, then y
    positions_east = sorted(
        positions_by_orientation["E"], key=lambda pos: (pos[1], pos[0])
    )
    while positions_east:
        pos = positions_east[0]
        track_length = 1
        y, x = pos
        while (
            (x + track_length) < width and
            track_generator.track[y, x + track_length] == DIRECTION_TO_VALUE["E"] and
            (y, x + track_length) in positions_east
        ):
            positions_east.remove((y, x + track_length))
            track_length += 1
        track_component = GridTrackStraight(
            name=f"Track_{x}_{y}_{track_length}xE",
            direction="E",
            translation=(
                (x - width / 2) * TRACK_SCALE,
                (-y + height / 2) * TRACK_SCALE,
                0.0,
            ),
            length=track_length,
        )
        positions_east.remove((y, x))
        group_track.children.append(track_component)

    # South: sort by increasing y, then x
    positions_south = sorted(
        positions_by_orientation["S"], key=lambda pos: (pos[0], pos[1])
    )
    while positions_south:
        pos = positions_south[0]
        track_length = 1
        y, x = pos
        while (
            (y + track_length) < height and
            track_generator.track[y + track_length, x] == DIRECTION_TO_VALUE["S"] and
            (y + track_length, x) in positions_south
        ):
            positions_south.remove((y + track_length, x))
            track_length += 1
        track_component = GridTrackStraight(
            name=f"Track_{x}_{y}_{track_length}xS",
            direction="S",
            translation=(
                (x - width / 2) * TRACK_SCALE,
                (-y + height / 2) * TRACK_SCALE,
                0.0,
            ),
            length=track_length,
        )
        positions_south.remove((y, x))
        group_track.children.append(track_component)
    # West: sort by decreasing x, then y
    positions_west = sorted(
        positions_by_orientation["W"], key=lambda pos: (-pos[1], pos[0])
    )
    while positions_west:
        pos = positions_west[0]
        track_length = 1
        y, x = pos
        while (
            (x - track_length) >= 0 and
            track_generator.track[y, x - track_length] == DIRECTION_TO_VALUE["W"] and
            (y, x - track_length) in positions_west
        ):
            positions_west.remove((y, x - track_length))
            track_length += 1
        track_component = GridTrackStraight(
            name=f"Track_{x}_{y}_{track_length}xW",
            direction="W",
            translation=(
                (x - width / 2) * TRACK_SCALE,
                (-y + height / 2) * TRACK_SCALE,
                0.0,
            ),
            length=track_length,
        )
        positions_west.remove((y, x))
        group_track.children.append(track_component)
    # North: sort by decreasing y, then x
    positions_north = sorted(
        positions_by_orientation["N"], key=lambda pos: (-pos[0], pos[1])
    )
    while positions_north:
        pos = positions_north[0]
        track_length = 1
        y, x = pos
        while (
            (y - track_length) >= 0 and
            track_generator.track[y - track_length, x] == DIRECTION_TO_VALUE["N"] and
            (y - track_length, x) in positions_north
        ):
            positions_north.remove((y - track_length, x))
            track_length += 1
        track_component = GridTrackStraight(
            name=f"Track_{x}_{y}_{track_length}xN",
            direction="N",
            translation=(
                (x - width / 2) * TRACK_SCALE,
                (-y + height / 2) * TRACK_SCALE,
                0.0,
            ),
            length=track_length,
        )
        positions_north.remove((y, x))
        group_track.children.append(track_component)
    
    # ==== Saving world ====
    world_str = world.get_string()
    with open(PATH_WBT, "w", encoding="utf-8") as f:
        f.write(world_str)
