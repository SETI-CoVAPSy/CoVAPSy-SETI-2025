"""
Generate a Webots world.
"""

import numpy as np

from pathlib import Path
from typing import Literal
from PIL import Image

from lib_track_generator import TrackGenerator, VALUE_TO_DIRECTION, DIRECTION_TO_VALUE
from lib_webots_world import World, CoVAPSyCar, Floor, BorderStraight, BorderCurve, ComponentBase, ComponentGroup

# =============================================
#  User parameters
# =============================================
MAP_WIDTH = 50  # in track units
MAP_HEIGHT = 30  # in track units
TRACK_MARGIN = min(MAP_WIDTH, MAP_HEIGHT)//5  # in track units
TRACK_SCALE = 2.0  # in meters
TRACK_GENERATION_STEPS = (MAP_WIDTH * MAP_HEIGHT)*5

PATH_WORLD = Path(__file__).parent / "Webots_SETI_gen" / "worlds"
PATH_WBT = PATH_WORLD / "CoVAPSy_SETI_2025_generated.wbt"
PATH_TRACK_IMAGE = PATH_WORLD / "generated_track_preview.png"
PATH_TEST_TRACK_IMAGE = Path(__file__).parent / "track_template_build_test.png" # Track template to import from
 
DO_TEST_TRACK = False  # If True, load track from test image instead of generating it

class GridTrackStraight(ComponentBase):
    """Straight track component."""

    def __init__(
            self, 
            name: str, 
            direction: Literal['N', 'E', 'S', 'W'],
            translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
        ) -> None:
        self.name = name
        track_red: BorderStraight
        track_green: BorderStraight
        if direction == "N":
            track_red = BorderStraight(
                f"{name}_Red",
                length=TRACK_SCALE,
                translation=(translation[0]-0.25*TRACK_SCALE, translation[1], translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(1.0, 0.0, 0.0)
            )
            track_green = BorderStraight(
                f"{name}_Green",
                length=TRACK_SCALE,
                translation=(translation[0]+0.25*TRACK_SCALE, translation[1], translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(0.0, 1.0, 0.0)
            )
        elif direction == "E":
            track_red = BorderStraight(
                f"{name}_Red",
                length=TRACK_SCALE,
                translation=(translation[0], translation[1]+0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0)
            )
            track_green = BorderStraight(
                f"{name}_Green",
                length=TRACK_SCALE,
                translation=(translation[0], translation[1]-0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0)
            )
        elif direction == "S":
            track_red = BorderStraight(
                f"{name}_Red",
                length=TRACK_SCALE,
                translation=(translation[0]+0.25*TRACK_SCALE, translation[1], translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(1.0, 0.0, 0.0)
            )
            track_green = BorderStraight(
                f"{name}_Green",
                length=TRACK_SCALE,
                translation=(translation[0]-0.25*TRACK_SCALE, translation[1], translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(0.0, 1.0, 0.0)
            )
        elif direction == "W":
            track_red = BorderStraight(
                f"{name}_Red",
                length=TRACK_SCALE,
                translation=(translation[0], translation[1]-0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0)
            )
            track_green = BorderStraight(
                f"{name}_Green",
                length=TRACK_SCALE,
                translation=(translation[0], translation[1]+0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0)
            )
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        self.track_components = ComponentGroup(
            [track_red, track_green],
            name=self.name
        )
    
    def get_string(self, indent: str = '  ') -> str:
        """Get string representation."""
        return self.track_components.get_string(indent=indent)


class GridTrackCurve(ComponentBase):
    """Curved track component."""

    # + 2 straight borders

    def __init__(
            self, 
            name: str, 
            direction_in: Literal['N', 'E', 'S', 'W'],
            direction_out: Literal['N', 'E', 'S', 'W'],
            translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
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
                translation=(translation[0], translation[1], translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, 3*np.pi/2),
                color=(1.0, 0.0, 0.0)
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0]+0.5*TRACK_SCALE, translation[1]-0.5*TRACK_SCALE, translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, 3*np.pi/2),
                color=(0.0, 1.0, 0.0)
            )
            straight_red1 = BorderStraight(
                f"{name}_Red_Straight1",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]-0.25*TRACK_SCALE, translation[1]-0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(1.0, 0.0, 0.0)
            )
            straight_red2 = BorderStraight(
                f"{name}_Red_Straight2",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]+0.25*TRACK_SCALE, translation[1]+0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0)
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_red1, straight_red2],
                name=self.name
            )
        elif direction_in == "S" and direction_out == "E":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0]+0.5*TRACK_SCALE, translation[1]+0.5*TRACK_SCALE, translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0)
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0], translation[1], translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0)
            )
            straight_green1 = BorderStraight(
                f"{name}_Green_Straight1",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]-0.25*TRACK_SCALE, translation[1]+0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(0.0, 1.0, 0.0)
            )
            straight_green2 = BorderStraight(
                f"{name}_Green_Straight2",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]+0.25*TRACK_SCALE, translation[1]-0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0)
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_green1, straight_green2],
                name=self.name
            )
        elif direction_in == "E" and direction_out == "S":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0], translation[1], translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, np.pi),
                color=(1.0, 0.0, 0.0)
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0]-0.5*TRACK_SCALE, translation[1]-0.5*TRACK_SCALE, translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, np.pi),
                color=(0.0, 1.0, 0.0)
            )
            straight_red1 = BorderStraight(
                f"{name}_Red_Straight1",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]+0.25*TRACK_SCALE, translation[1]-0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(1.0, 0.0, 0.0)
            )
            straight_red2 = BorderStraight(
                f"{name}_Red_Straight2",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]-0.25*TRACK_SCALE, translation[1]+0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0)
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_red1, straight_red2],
                name=self.name
            )
        elif direction_in == "W" and direction_out == "S":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0]+0.5*TRACK_SCALE, translation[1]-0.5*TRACK_SCALE, translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, 3*np.pi/2),
                color=(1.0, 0.0, 0.0)
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0], translation[1], translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, 3*np.pi/2),
                color=(0.0, 1.0, 0.0)
            )
            straight_green1 = BorderStraight(
                f"{name}_Green_Straight1",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]-0.25*TRACK_SCALE, translation[1]-0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(0.0, 1.0, 0.0)
            )
            straight_green2 = BorderStraight(
                f"{name}_Green_Straight2",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]+0.25*TRACK_SCALE, translation[1]+0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0)
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_green1, straight_green2],
                name=self.name
            )
        elif direction_in == "S" and direction_out == "W":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0], translation[1], translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(1.0, 0.0, 0.0)
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0]-0.5*TRACK_SCALE, translation[1]+0.5*TRACK_SCALE, translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(0.0, 1.0, 0.0)
            )
            straight_red1 = BorderStraight(
                f"{name}_Red_Straight1",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]+0.25*TRACK_SCALE, translation[1]+0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(1.0, 0.0, 0.0)
            )
            straight_red2 = BorderStraight(
                f"{name}_Red_Straight2",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]-0.25*TRACK_SCALE, translation[1]-0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0)
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_red1, straight_red2],
                name=self.name
            )
        elif direction_in == "N" and direction_out == "W":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0]-0.5*TRACK_SCALE, translation[1]-0.5*TRACK_SCALE, translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, np.pi),
                color=(1.0, 0.0, 0.0)
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0], translation[1], translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, np.pi),
                color=(0.0, 1.0, 0.0)
            )
            straight_green1 = BorderStraight(
                f"{name}_Green_Straight1",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]+0.25*TRACK_SCALE, translation[1]-0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(0.0, 1.0, 0.0)
            )
            straight_green2 = BorderStraight(
                f"{name}_Green_Straight2",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]-0.25*TRACK_SCALE, translation[1]+0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0)
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_green1, straight_green2],
                name=self.name
            )
        elif direction_in == "E" and direction_out == "N":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0]-0.5*TRACK_SCALE, translation[1]+0.5*TRACK_SCALE, translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(1.0, 0.0, 0.0)
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0], translation[1], translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(0.0, 1.0, 0.0)
            )
            straight_green1 = BorderStraight(
                f"{name}_Green_Straight1",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]+0.25*TRACK_SCALE, translation[1]+0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(0.0, 1.0, 0.0)
            )
            straight_green2 = BorderStraight(
                f"{name}_Green_Straight2",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]-0.25*TRACK_SCALE, translation[1]-0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0)
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_green1, straight_green2],
                name=self.name
            )
        elif direction_in == "W" and direction_out == "N":
            curve_red = BorderCurve(
                f"{name}_Red",
                translation=(translation[0], translation[1], translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0)
            )
            curve_green = BorderCurve(
                f"{name}_Green",
                translation=(translation[0]+0.5*TRACK_SCALE, translation[1]+0.5*TRACK_SCALE, translation[2]+0.1),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(0.0, 1.0, 0.0)
            )
            straight_red1 = BorderStraight(
                f"{name}_Red_Straight1",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]-0.25*TRACK_SCALE, translation[1]+0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, np.pi/2),
                color=(1.0, 0.0, 0.0)
            )
            straight_red2 = BorderStraight(
                f"{name}_Red_Straight2",
                length=0.5*TRACK_SCALE,
                translation=(translation[0]+0.25*TRACK_SCALE, translation[1]-0.25*TRACK_SCALE, translation[2]),
                rotation=(0.0, 0.0, 1.0, 0.0),
                color=(1.0, 0.0, 0.0)
            )
            self.track_components = ComponentGroup(
                [curve_red, curve_green, straight_red1, straight_red2],
                name=self.name
            )    
        else:
            self.track_components = ComponentGroup([], name=self.name)
        

    def get_string(self, indent: str = '  ') -> str:
        """Get string representation."""
        return self.track_components.get_string(indent=indent)


if __name__ == "__main__":
    if not PATH_WORLD.exists():
        raise FileNotFoundError(f"World parent path not found: {PATH_WORLD}")

    # ==== World definition ====
    world = World(world_path=str(PATH_WBT))

    # ==== Track ====
    if not DO_TEST_TRACK:
        # === Generate track ===
        track_generator = TrackGenerator(
            width=MAP_WIDTH, height=MAP_HEIGHT, margin=TRACK_MARGIN
        )
        for _ in range(TRACK_GENERATION_STEPS):
            track_generator.track_gen_step()
    
    else: # === Load test track ===
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
    car1 = CoVAPSyCar(
        name="CoVAPSy_Car_1",
        controller="<extern>",
        color=(1.0, 0.0, 0.0),  # Red
        translation=(0.0, -1.0, 0),
    )
    world.components.append(car1)

    # === Floor ===
    floor_size = (float(max(MAP_WIDTH, MAP_HEIGHT)) + 1) * TRACK_SCALE 
    floor = Floor(name="Ground_Floor", size=(floor_size, floor_size))
    world.components.append(floor)

    # === Track components ===
    height, width = track_generator.track.shape
    for y in range(height):
        for x in range(width):
            cell_value = track_generator.track[y, x]
            direction = VALUE_TO_DIRECTION[cell_value]
            if direction is None:
                continue  # No track

            # # Curves
            if (# N->E
                direction == 'E' and
                y < height - 1 and
                track_generator.track[y+1, x] == DIRECTION_TO_VALUE['N']
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in='N',
                    direction_out='E',
                    translation=((x - width / 2) * TRACK_SCALE, (-y + height / 2) * TRACK_SCALE, 0.0)
                )
                world.components.append(track_component)
                continue
            elif (# S -> E
                direction == 'E' and
                y > 0 and
                track_generator.track[y-1, x] == DIRECTION_TO_VALUE['S']
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in='S',
                    direction_out='E',
                    translation=((x - width / 2) * TRACK_SCALE, (-y + height / 2) * TRACK_SCALE, 0.0)
                )
                world.components.append(track_component)
                continue
            elif (# E -> S
                direction == 'S' and
                x > 0 and
                track_generator.track[y, x-1] == DIRECTION_TO_VALUE['E']
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in='E',
                    direction_out='S',
                    translation=((x - width / 2) * TRACK_SCALE, (-y + height / 2) * TRACK_SCALE, 0.0)
                )
                world.components.append(track_component)
                continue
            elif (# W -> S
                direction == 'S' and
                x < width - 1 and
                track_generator.track[y, x+1] == DIRECTION_TO_VALUE['W']
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in='W',
                    direction_out='S',
                    translation=((x - width / 2) * TRACK_SCALE, (-y + height / 2) * TRACK_SCALE, 0.0)
                )
                world.components.append(track_component)
                continue
            elif (# S -> W
                direction == 'W' and
                y > 0 and
                track_generator.track[y-1, x] == DIRECTION_TO_VALUE['S']
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in='S',
                    direction_out='W',
                    translation=((x - width / 2) * TRACK_SCALE, (-y + height / 2) * TRACK_SCALE, 0.0)
                )
                world.components.append(track_component)
                continue
            elif (# N -> W
                direction == 'W' and
                y < height - 1 and
                track_generator.track[y+1, x] == DIRECTION_TO_VALUE['N']
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in='N',
                    direction_out='W',
                    translation=((x - width / 2) * TRACK_SCALE, (-y + height / 2) * TRACK_SCALE, 0.0)
                )
                world.components.append(track_component)
                continue
            elif (# E -> N
                direction == 'N' and
                x > 0 and
                track_generator.track[y, x-1] == DIRECTION_TO_VALUE['E']
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in='E',
                    direction_out='N',
                    translation=((x - width / 2) * TRACK_SCALE, (-y + height / 2) * TRACK_SCALE, 0.0)
                )
                world.components.append(track_component)
                continue
            elif (# W -> N
                direction == 'N' and
                x < width - 1 and
                track_generator.track[y, x+1] == DIRECTION_TO_VALUE['W']
            ):
                track_component = GridTrackCurve(
                    name=f"Track_Curve_{x}_{y}",
                    direction_in='W',
                    direction_out='N',
                    translation=((x - width / 2) * TRACK_SCALE, (-y + height / 2) * TRACK_SCALE, 0.0)
                )
                world.components.append(track_component)
                continue

            # Straights
            track_component = GridTrackStraight(
                name=f"Track_{x}_{y}",
                direction=direction,
                translation=((x - width / 2) * TRACK_SCALE, (-y + height / 2) * TRACK_SCALE, 0.0)
            )
            world.components.append(track_component)

    # ==== Saving world ====
    world_str = world.get_string()
    with open(PATH_WBT, "w", encoding="utf-8") as f:
        f.write(world_str)
