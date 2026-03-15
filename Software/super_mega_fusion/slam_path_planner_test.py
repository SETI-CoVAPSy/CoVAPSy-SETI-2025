from path_planner import (
    occupancy_grid_preprocess,
    make_graph_and_flow_from_occupancy_grid,
    pose_to_pixel,
    naive_path_planning,
    naive_position_prediction,
    super_mega_fusion_path_planning,
    trajectory_to_positions,
    draw_figure,
    TrackWallDirection,
    Command,
    Pose2D,
    OccupancyGridLabelled,
    SegmentationLabels,
)
import numpy as np
from pathlib import Path
from typing import cast


# ====================================================
#  Tests
# ====================================================
if __name__ == "__main__":
    from pathlib import Path
    from matplotlib import pyplot as plt
    from PIL import Image

    # ====== Parameters ======

    WHEELBASE_M = 0.5  # m distance between front and rear axles
    PIXELS_PER_METER = 80  # pixels for one meter
    COLLISION_RADIUS_M = 0.05  # radius of collision area around the obstacles
    OPPONENT_COLLISION_RADIUS_M = 0.1  # m, collision radius for opponent avoidance
    OPPONENT_SPEED_MPS = 0.5  # meters per second, for naive position prediction
    SAMPLE_FREQUENCY_HZ = 4.0  # how often to sample along the path (Hz)
    COMMAND_TRAJECTORY_SUBSTEPS = (
        5  # how many trajectory points to precompute for each command
    )
    BACKTRACK_COUNT = 1  # how many steps to backtrack during path planning

    # Planning
    sfh, cts, wb = SAMPLE_FREQUENCY_HZ, COMMAND_TRAJECTORY_SUBSTEPS, WHEELBASE_M
    PLANNING_COMMANDS: list[Command] = [  #
        Command(0.0, 1.0, sfh, cts, wb),  # straight
        Command(np.radians(15), 0.8, sfh, cts, wb),  # slight left
        Command(np.radians(-15), 0.8, sfh, cts, wb),  # slight right
        Command(0.0, 0.3, sfh, cts, wb),  # straight slow
        Command(np.radians(30), 0.5, sfh, cts, wb),  # left
        Command(np.radians(-30), 0.5, sfh, cts, wb),  # right
        Command(np.radians(60), 0.3, sfh, cts, wb),  # sharp left
        Command(np.radians(-60), 0.3, sfh, cts, wb),  # sharp right
    ]
    MAX_ITERATIONS = 200_000  # maximum iterations for the path planning search
    TRAJECTORY_PLAN_LENGTH = 70  # default number of trajectory points to plan

    CLOSEST_PATH_BIAS_STRENGTH = 10  # softmax temperature scale for A* reference bias
    COMMAND_PRIORITY_BIAS_STRENGTH = (
        0  # softmax scale favouring lower-index (faster) commands
    )

    # Start positions
    position_start_seti = Pose2D(x=-0.7, y=1.9, theta=np.pi / 7)
    position_start_oponents = [
        Pose2D(x=-0.2, y=2.1, theta=0.0),
        Pose2D(x=0.3, y=1.8, theta=0.0),
    ]
    position_target = Pose2D(x=-0.8, y=-1.8, theta=0.0)

    # Load occupancy grid from image
    print("Loading occupancy grid...")
    img = Image.open(Path(__file__).parent / "slam_resources" / "carte.png").convert(
        "RGB"
    )
    img_array = np.array(img, dtype=np.uint8)  # shape (H, W, 3)

    # Convert to occupancy grid with labels
    print("Converting to occupancy grid with labels...")
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    occupancy_grid_labelled = cast(
        OccupancyGridLabelled,
        np.full(r.shape, SegmentationLabels.MISC_OBSTACLE.value, dtype=np.uint8),
    )
    occupancy_grid_labelled[(r == 255) & (g == 255) & (b == 255)] = (
        SegmentationLabels.FREE.value
    )
    occupancy_grid_labelled[(r == 255) & (g == 0) & (b == 0)] = (
        SegmentationLabels.WALL_RED.value
    )
    occupancy_grid_labelled[(r == 0) & (g == 255) & (b == 0)] = (
        SegmentationLabels.WALL_GREEN.value
    )

    if False:
        print("Unique values in occupancy grid:", np.unique(occupancy_grid_labelled))

    # Test occupancy_grid_preprocess
    print("Testing occupancy_grid_preprocess...")
    occupancy_grid_labelled = occupancy_grid_preprocess(
        occupancy_grid_labelled,
        dilation_radius=int(np.ceil(COLLISION_RADIUS_M * PIXELS_PER_METER)),
    )
    if False:
        plt.imshow(occupancy_grid_labelled, cmap="gray")
        plt.title("Preprocessed Occupancy Grid")
        plt.show()

    # Test make_graph_from_occupancy_grid
    print("Testing make_graph_from_occupancy_grid...")
    graph, flow_field = make_graph_and_flow_from_occupancy_grid(
        occupancy_grid_labelled, TrackWallDirection.RED_RIGHT_GREEN_LEFT
    )
    if False:
        print(
            f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
        )

    H, W = occupancy_grid_labelled.shape
    cell_start_seti = pose_to_pixel(position_start_seti, PIXELS_PER_METER, W, H)
    cell_start_opponents = [
        pose_to_pixel(pos, PIXELS_PER_METER, W, H) for pos in position_start_oponents
    ]
    cell_target = pose_to_pixel(position_target, PIXELS_PER_METER, W, H)
    cell_origin = pose_to_pixel(Pose2D(x=0, y=0, theta=0), PIXELS_PER_METER, W, H)

    # Test naive_path_planning
    print("Testing naive_path_planning...")
    path_seti = naive_path_planning(graph, cell_start_seti, cell_target)
    path_opponents = [
        naive_path_planning(graph, cell_start, cell_target)
        for cell_start in cell_start_opponents
    ]
    # Test naive_position_prediction
    print("Testing naive_position_prediction...")
    speed_pps = OPPONENT_SPEED_MPS * PIXELS_PER_METER  # pixels per second
    positions_seti = naive_position_prediction(
        path_seti,
        speed_pps,
        PIXELS_PER_METER,
        sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
        grid_shape=occupancy_grid_labelled.shape,
    )
    positions_opponents = [
        naive_position_prediction(
            path,
            speed_pps,
            pixels_per_meter=PIXELS_PER_METER,
            sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
            grid_shape=occupancy_grid_labelled.shape,
        )
        for path in path_opponents
    ]

    # Test super_mega_fusion_path_planning
    print("Testing super_mega_fusion_path_planning...")
    seti_trajectory = super_mega_fusion_path_planning(
        occupancy_grid_labelled,
        position_start_seti,
        PLANNING_COMMANDS,
        OPPONENT_COLLISION_RADIUS_M,
        TRAJECTORY_PLAN_LENGTH,
        PIXELS_PER_METER,
        estimated_positions_seti=positions_seti,
        estimated_positions_opponents=positions_opponents,
        max_iterations=MAX_ITERATIONS,
        backtrack_count=BACKTRACK_COUNT,
        closest_path_bias_strength=CLOSEST_PATH_BIAS_STRENGTH,
        command_priority_bias_strength=COMMAND_PRIORITY_BIAS_STRENGTH,
    )

    # Display results
    if True:
        # expand the coarse endpoint trajectory into a dense set of samples so
        # the drawn path appears smooth rather than zig‑zaggy
        smooth_seti = trajectory_to_positions(seti_trajectory)

        draw_figure(
            occupancy_grid_labelled,
            PIXELS_PER_METER,
            COLLISION_RADIUS_M,
            positions_naive_seti=positions_seti,
            positions_seti=smooth_seti,
            positions_opponents=positions_opponents,
            start_position_seti=position_start_seti,
            start_positions_opponents=position_start_oponents,
            target_pose=position_target,
            flow_field=flow_field,
            # misc_scatter_points=[],
            scale_icons=6,
            title="Path planner",
        )
