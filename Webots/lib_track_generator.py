"""
Library for generating tracks in Webots simulations.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal, TypedDict, Callable
from PIL import Image


Track = NDArray[np.int8] # Shape: (height, width), values: -1=empty, 0-3=direction
DIRECTION_TO_VALUE: dict[Literal['N', 'E', 'S', 'W', None], int] = {
    'N': 0,
    'E': 1,
    'S': 2,
    'W': 3,
    None: -1
}
VALUE_TO_DIRECTION: dict[int, Literal['N', 'E', 'S', 'W', None]] = {
    0: 'N',
    1: 'E',
    2: 'S',
    3: 'W',
    -1: None
}

class TrackElementaryOperation(TypedDict):
    """Elementary operation on a local part of the track.
    Always assumes [x,y] to be east."""
    condition: Callable[[Track, int, int], bool] # (track, y, x) -> bool
    operation: Callable[[Track, int, int], None] # (track, y, x) -> None

class TrackGenerator:
    """Class to generate tracks for Webots simulations."""

    def __init__(self, width: int, height: int, margin: int = 0) -> None:
        """Class to generate tracks for Webots simulations.
        
        Args:
            width: Width of the track in cells.
            height: Height of the track in cells.
            margin: Margin around the border for initial track.
        """
        self.track = np.full((height, width), -1, dtype=np.int8) # Empty track
        # Add track (loop around the border)
        for x in range(margin, width - margin): # Top row
            self.track[margin, x] = DIRECTION_TO_VALUE['E']
        for y in range(margin, height - margin): # Right column
            self.track[y, width - margin - 1] = DIRECTION_TO_VALUE['S']
        for x in range(width - margin - 1, margin - 1, -1): # Bottom row
            self.track[height - margin - 1, x] = DIRECTION_TO_VALUE['W']
        for y in range(height - margin - 1, margin, -1): # Left column
            self.track[y, margin] = DIRECTION_TO_VALUE['N']
        
        # ====================================================
        #  Allowed operations
        # ====================================================
        self._operations: list[TrackElementaryOperation] = []
        
        # LEGEND:
        # >v<^ : Input directions (East, South, West, North)
        # →↑←↓ : Directions (East, North, West, South)
        # o    : Empty cell (None)
        # ?    : Any cell (including empty)
        # i    : No incoming (never input)

        # ----------------------------------------------------
        # >→ -> v→ (Inserted u-turn)
        # oo    →↑
        def operation_eeoo_sene_condition(track: Track, y: int, x: int) -> bool:
            h, w = track.shape
            return (
                x < w - 1 and y < h - 1 and
                track[y,   x]   == DIRECTION_TO_VALUE['E'] and
                track[y,   x+1] == DIRECTION_TO_VALUE['E'] and
                track[y+1, x]   == DIRECTION_TO_VALUE[None] and
                track[y+1, x+1] == DIRECTION_TO_VALUE[None]
            )
        def operation_eeoo_sene_operation(track: Track, y: int, x: int) -> None:
            track[y,   x]   = DIRECTION_TO_VALUE['S']
            track[y+1, x]   = DIRECTION_TO_VALUE['E']
            track[y+1, x+1] = DIRECTION_TO_VALUE['N']
        self._operations.append(
            TrackElementaryOperation(
                condition=operation_eeoo_sene_condition,
                operation=operation_eeoo_sene_operation
            )
        )

        # ----------------------------------------------------
        # >↑ - > v↑ (u-turn extension)
        # oo     →↑
        def operation_enoo_senn_condition(track: Track, y: int, x: int) -> bool:
            h, w = track.shape
            return (
                x < w - 1 and y < h - 1 and
                track[y,   x]   == DIRECTION_TO_VALUE['E'] and
                track[y,   x+1] == DIRECTION_TO_VALUE['N'] and
                track[y+1, x]   == DIRECTION_TO_VALUE[None] and
                track[y+1, x+1] == DIRECTION_TO_VALUE[None]
            )
        def operation_enoo_senn_operation(track: Track, y: int, x: int) -> None:
            track[y,   x]   = DIRECTION_TO_VALUE['S']
            track[y+1, x]   = DIRECTION_TO_VALUE['E']
            track[y+1, x+1] = DIRECTION_TO_VALUE['N']            
        self._operations.append(
            TrackElementaryOperation(
                condition=operation_enoo_senn_condition,
                operation=operation_enoo_senn_operation
            )
        )
        
        # # ----------------------------------------------------
        # # >ii -> v47 (3x3 transpose) (Not used)
        # # i?i    258
        # # ii→    36→
        # def operation_3p3transpose_condition(track: Track, y: int, x: int) -> bool:
        #     h, w = track.shape
        #     return (
        #         x < w - 2 and y < h - 2 and
        #         track[y,   x] == DIRECTION_TO_VALUE['E'] and
        #         track[y+2, x+2] == DIRECTION_TO_VALUE['E'] and
        #         (y == 0 or track[y-1, x+1]!= DIRECTION_TO_VALUE['S']) and
        #         (y == 0 or track[y-1, x+2]!= DIRECTION_TO_VALUE['S']) and
        #         (x+2 == w - 1 or track[y,   x+2]!= DIRECTION_TO_VALUE['W']) and
        #         (x+2 == w - 1 or track[y+1, x+2]!= DIRECTION_TO_VALUE['W']) and
        #         (y+2 == h - 1 or track[y+2, x+2]!= DIRECTION_TO_VALUE['W']) and
        #         (y+2 == h - 1 or track[y+2, x+2]!= DIRECTION_TO_VALUE['N']) and
        #         (y+2 == h - 1 or track[y+2, x+1]!= DIRECTION_TO_VALUE['N']) and
        #         (y+2 == h - 1 or track[y+2, x]  != DIRECTION_TO_VALUE['N']) and
        #         (x == 0 or track[y+2, x-1]!= DIRECTION_TO_VALUE['E']) and
        #         (x == 0 or track[y+1, x-1]!= DIRECTION_TO_VALUE['E'])
        #     )
        # def operation_3p3transpose_operation(track: Track, y: int, x: int) -> None:
        #     local_transposed = track[y:y+3, x:x+3].T.copy()
        #     # Adjust directions
        #     mask_north = (local_transposed == DIRECTION_TO_VALUE['N'])
        #     mask_east  = (local_transposed == DIRECTION_TO_VALUE['E'])
        #     mask_south = (local_transposed == DIRECTION_TO_VALUE['S'])
        #     mask_west  = (local_transposed == DIRECTION_TO_VALUE['W'])
        #     local_transposed[mask_north] = DIRECTION_TO_VALUE['E']
        #     local_transposed[mask_east]  = DIRECTION_TO_VALUE['S']
        #     local_transposed[mask_south] = DIRECTION_TO_VALUE['W']
        #     local_transposed[mask_west]  = DIRECTION_TO_VALUE['N']
        #     track[y:y+3, x:x+3] = local_transposed
        #     track[y+2, x+2] = DIRECTION_TO_VALUE['E'] # Ensure the output direction is east
        # self._operations.append(
        #     TrackElementaryOperation(
        #         condition=operation_3p3transpose_condition,
        #         operation=operation_3p3transpose_operation
        #     )
        # )
        
    def track_gen_step(self) -> None:
        """Perform a single step of track generation."""
        # Rotate track randomly
        new_track = self.get_rotated(self.track, rotation_steps=np.random.choice([0, 1, 2, 3]))
        # Flip track randomly
        if np.random.choice([True, False]):
            axis = np.random.choice([0, 1]) # 0=vertical, 1=horizontal
            new_track = self.get_flipped(new_track, axis=axis)
        relevant_positions = np.argwhere(new_track == DIRECTION_TO_VALUE['E']) # All positions that are relevant. Here: towards east

        # Select a random filled position
        height, width = new_track.shape
        y, x = relevant_positions[np.random.choice(len(relevant_positions))]

        # Get all possible operations at this position
        possible_operations = [op for op in self._operations if op['condition'](new_track, y, x)]
        if not possible_operations:
            return  # No operation can be applied
        
        # Select a random operation
        operation_index = np.random.choice(range(len(possible_operations)))
        # Apply the operation
        possible_operations[operation_index]['operation'](new_track, y, x)
        
        # Update the track (rotate back)
        self.track = new_track

    @staticmethod
    def get_rotated(track: Track, rotation_steps: int) -> Track:
        """Get rotated track 90 degrees clockwise.
        
        Args:
            track: The track to rotate.
            rotation_steps: Number of 90 degree clockwise rotations.
        """
        new_track = track.copy()
        if rotation_steps % 4 == 0:
            return new_track
        elif rotation_steps % 4 == 1: # 90 degrees
            rotated_track = np.rot90(new_track, k=3)
            # For all non -1, add 1 mod 4
            mask = (rotated_track != -1)
            rotated_track[mask] = (rotated_track[mask] + 1) % 4
            return rotated_track
        elif rotation_steps % 4 == 2: # 180 degrees
            rotated_track = np.rot90(new_track, k=2)
            # For all non -1, add 2 mod 4
            mask = (rotated_track != -1)
            rotated_track[mask] = (rotated_track[mask] + 2) % 4
            return rotated_track
        else: # rotation_steps % 4 == 3: # 270 degrees
            rotated_track = np.rot90(new_track, k=1)
            # For all non -1, add 3 mod 4
            mask = (rotated_track != -1)
            rotated_track[mask] = (rotated_track[mask] + 3) % 4
            return rotated_track
    
    @staticmethod
    def get_flipped(track: Track, axis: int) -> Track:
        """Get flipped track.
        
        Args:
            track: The track to flip.
            axis: Axis to flip around. 0=vertical, 1=horizontal.
        """
        flipped_track = np.flip(track.copy(), axis=axis)
        if axis == 0: # Vertical flip
            # For all non -1, N<->S
            mask_north = (flipped_track == DIRECTION_TO_VALUE['N'])
            mask_south = (flipped_track == DIRECTION_TO_VALUE['S'])
            flipped_track[mask_north] = DIRECTION_TO_VALUE['S']
            flipped_track[mask_south] = DIRECTION_TO_VALUE['N']
        else: # Horizontal flip
            # For all non -1, E<->W
            mask_east = (flipped_track == DIRECTION_TO_VALUE['E'])
            mask_west = (flipped_track == DIRECTION_TO_VALUE['W'])
            flipped_track[mask_east] = DIRECTION_TO_VALUE['W']
            flipped_track[mask_west] = DIRECTION_TO_VALUE['E']
        return flipped_track

    @staticmethod
    def save_track_matrix_image(
        track: Track,
        file_path: str,
        color_none: tuple[int, int, int] = (255, 255, 255), # WHITE
        color_east: tuple[int, int, int] = (230, 23, 23),   # RED
        color_south: tuple[int, int, int] = (126, 230, 23), # GREEN
        color_west: tuple[int, int, int] = (23, 230, 230),  # CYAN 
        color_north: tuple[int, int, int] = (126, 23, 230), # PURPLE
    ) -> None:
        """Save the track to a file.
        
        Args:
            file_path: The file_path to save the track to.
            color_east: RGB color for east direction.
            color_south: RGB color for south direction.
            color_west: RGB color for west direction.
            color_north: RGB color for north direction.
        """
        height, width = track.shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        # None 
        mask_none = (track == -1)
        image[mask_none] = color_none
        # East
        mask_east = (track == DIRECTION_TO_VALUE['E'])
        image[mask_east] = color_east
        # South
        mask_south = (track == DIRECTION_TO_VALUE['S'])
        image[mask_south] = color_south
        # West
        mask_west = (track == DIRECTION_TO_VALUE['W'])
        image[mask_west] = color_west
        # North
        mask_north = (track == DIRECTION_TO_VALUE['N'])
        image[mask_north] = color_north

        # Save image
        img = Image.fromarray(image, 'RGB')
        img.save(file_path)
    
    @staticmethod
    def save_track_bitmap_image(
        track: Track,
        file_path: str,
        cell_size: int = 5,
        cell_none_color: tuple[int, int, int] = (255, 255, 255), # White
        cell_north_color: tuple[int, int, int] = (126, 23, 230), # PURPLE
        cell_east_color: tuple[int, int, int] = (230, 23, 23),   # RED
        cell_south_color: tuple[int, int, int] = (126, 230, 23), # GREEN
        cell_west_color: tuple[int, int, int] = (23, 230, 230),  # CYAN
        cell_north_template: NDArray[np.bool_] = np.array([[False, False, True, False, False],
                                                           [False, True,  True, True,  False],
                                                           [True,  False, True, False, True ],
                                                           [False, False, True, False, False],
                                                           [False, False, True, False, False]
                                                          ], dtype=np.bool_),
    ) -> None:
        """Save the track to a bitmap image file.
        
        Args:
            track: The track to save.
            file_path: The file_path to save the track to.
            cell_size: Size of each cell in pixels.
            cell_north: Bitmap for north direction cell.
            cell_none_color: RGB color for none direction cell.
        """
        height, width = track.shape
        image = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)
        # Create cell bitmaps for each direction
        cell_north = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        cell_north[:] = np.where(cell_north_template[..., None], cell_north_color, cell_none_color)

        cell_east = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        cell_east[:] = np.where(np.rot90(cell_north_template, k=3)[..., None], cell_east_color, cell_none_color)

        cell_south = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        cell_south[:] = np.where(np.rot90(cell_north_template, k=2)[..., None], cell_south_color, cell_none_color)

        cell_west = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        cell_west[:] = np.where(np.rot90(cell_north_template, k=1)[..., None], cell_west_color, cell_none_color)

        # Fill the image
        for i in range(height):
            for j in range(width):
                y0 = i * cell_size
                x0 = j * cell_size
                direction = VALUE_TO_DIRECTION[track[i, j]]
                if direction == 'N':
                    image[y0:y0+cell_size, x0:x0+cell_size] = cell_north
                elif direction == 'E':
                    image[y0:y0+cell_size, x0:x0+cell_size] = cell_east
                elif direction == 'S':
                    image[y0:y0+cell_size, x0:x0+cell_size] = cell_south
                elif direction == 'W':
                    image[y0:y0+cell_size, x0:x0+cell_size] = cell_west
                else: # None
                    image[y0:y0+cell_size, x0:x0+cell_size] = cell_none_color
        # Save image
        Image.fromarray(image, 'RGB').save(file_path)

    @staticmethod
    def load_from_matrix_image(
        image_data: NDArray[np.uint8],
        color_north: tuple[int, int, int] = (126, 23, 230), # PURPLE
        color_east: tuple[int, int, int] = (230, 23, 23),   # RED
        color_south: tuple[int, int, int] = (126, 230, 23), # GREEN
        color_west: tuple[int, int, int] = (23, 230, 230),  # CYAN
        ) -> "TrackGenerator":
        """Load track from a matrix image."""
        height, width, _ = image_data.shape
        track = np.full((height, width), -1, dtype=np.int8)
        for i in range(height):
            for j in range(width):
                pixel = tuple(image_data[i, j])
                if pixel == color_north:
                    track[i, j] = DIRECTION_TO_VALUE['N']
                elif pixel == color_east:
                    track[i, j] = DIRECTION_TO_VALUE['E']
                elif pixel == color_south:
                    track[i, j] = DIRECTION_TO_VALUE['S']
                elif pixel == color_west:
                    track[i, j] = DIRECTION_TO_VALUE['W']
                else:
                    track[i, j] = DIRECTION_TO_VALUE[None]
        track_generator = TrackGenerator(width=width, height=height)
        track_generator.track = track
        return track_generator

if __name__ == '__main__':
    # =================================================
    #  Example usage
    # =================================================
    track_generator = TrackGenerator(width=20, height=15, margin=2)
    # track_generator.track[0,0] = DIRECTION_TO_VALUE['S']

    for step in range(1000):
        if step % 100 == 0:
            filename = f'track_matrix_step_{step}.png'
            TrackGenerator.save_track_bitmap_image(
                track_generator.track,
                file_path=filename
            )
            print(f'Saved {filename}')
        track_generator.track_gen_step()
    
    # Final save
    TrackGenerator.save_track_bitmap_image(
        track_generator.track,
        file_path='track_matrix_final.png'
    )
    # Flipped final save
    TrackGenerator.save_track_bitmap_image(
        TrackGenerator.get_flipped(track_generator.track, axis=0),
        file_path='track_matrix_final_flipped_v.png'
    )
    TrackGenerator.save_track_bitmap_image(
        TrackGenerator.get_flipped(track_generator.track, axis=1),
        file_path='track_matrix_final_flipped_h.png'
    )
    print('Saved final track images.')
    