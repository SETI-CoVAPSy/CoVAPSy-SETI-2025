from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    sim_time_arg = {"use_sim_time": True}
    waypoints_x = [
        -3.67578, -4.25027, -4.00572, -2.98572, -1.75572, -0.71572,
        0.19428, -0.07572, -0.17572, 0.877861, 2.00204, 2.38204,
        3.18204, 2.82204, 3.20204, 2.95459, 2.07791, 1.81791,
        2.34711, 4.33711, 5.13507, 5.6936, 5.6936, 5.0386,
        3.3986, 0.0886, -2.07502, -2.28502, -3.84042,
    ]
    waypoints_y = [
        -1.79182, -2.97463, -3.97463, -4.20463, -3.96463, -2.88463,
        -1.76463, -0.45463, 0.83537, 1.88882, 3.01298, 3.16298,
        2.67802, 0.58802, -1.90198, -3.12453, -4.00121, -4.47121,
        -5.18839, -5.18839, -4.99043, -3.2919, 3.8681, 5.00256,
        5.16256, 5.16256, 2.99909, -0.36091, -1.81742,
    ]

    webots_host_arg = DeclareLaunchArgument(
        "webots_host",
        default_value="192.168.65.254",
        description="Host/IP where Webots is running",
    )
    webots_port_arg = DeclareLaunchArgument(
        "webots_port",
        default_value="1234",
        description="Webots remote extern controller TCP port",
    )
    webots_robot_arg = DeclareLaunchArgument(
        "webots_robot_name",
        default_value="TT02_jaune_python",
        description="Webots robot name waiting for extern controller",
    )
    run_mpc_arg = DeclareLaunchArgument(
        "run_mpc",
        default_value="true",
        description="Start MPC node in addition to the driver",
    )
    run_plotter_arg = DeclareLaunchArgument(
        "run_plotter",
        default_value="true",
        description="Start realtime matplotlib plotter for MPC diagnostics",
    )
    plotter_scan_angle_offset_deg_arg = DeclareLaunchArgument(
        "plotter_scan_angle_offset_deg",
        default_value="0.0",
        description="Angular offset (deg) applied to lidar points in plotter",
    )
    plotter_scan_mirror_arg = DeclareLaunchArgument(
        "plotter_scan_mirror",
        default_value="true",
        description="Mirror lidar scan angles in plotter",
    )
    plotter_scan_reverse_arg = DeclareLaunchArgument(
        "plotter_scan_reverse",
        default_value="false",
        description="Reverse lidar scan order in plotter",
    )
    plotter_lidar_offset_x_arg = DeclareLaunchArgument(
        "plotter_lidar_offset_x",
        default_value="0.2",
        description="Lidar sensor x-offset from car center in plotter (m)",
    )
    plotter_lidar_offset_y_arg = DeclareLaunchArgument(
        "plotter_lidar_offset_y",
        default_value="0.0",
        description="Lidar sensor y-offset from car center in plotter (m)",
    )

    set_webots_url = SetEnvironmentVariable(
        name="WEBOTS_CONTROLLER_URL",
        value=[
            "tcp://",
            LaunchConfiguration("webots_host"),
            ":",
            LaunchConfiguration("webots_port"),
            "/",
            LaunchConfiguration("webots_robot_name"),
        ],
    )

    driver_node = Node(
        package="tt02_driver",
        executable="driver",
        name="tt02_driver",
        parameters=[{"target": "simulation"}, sim_time_arg],
        output="screen",
    )

    mpc_node = Node(
        package="tt02_driver",
        executable="mpc",
        name="tt02_mpc",
        parameters=[
            sim_time_arg,
            {"debug_decisions": True},
            {"wp_reached_distance": 0.6},
            {"q_ey": 20.0},
            {"r_steer": 0.05},
            {"wp_pass_margin": 0.0},
            {"max_steering_angle_deg": 20.0},
            {"waypoints_x": waypoints_x},
            {"waypoints_y": waypoints_y},
            {"scan_angle_offset_deg": LaunchConfiguration("plotter_scan_angle_offset_deg")},
            {
                "scan_mirror": ParameterValue(
                    LaunchConfiguration("plotter_scan_mirror"), value_type=bool
                )
            },
            {
                "scan_reverse": ParameterValue(
                    LaunchConfiguration("plotter_scan_reverse"), value_type=bool
                )
            },
            {"lidar_offset_x": LaunchConfiguration("plotter_lidar_offset_x")},
            {"lidar_offset_y": LaunchConfiguration("plotter_lidar_offset_y")},
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("run_mpc")),
    )

    plotter_node = Node(
        package="tt02_driver",
        executable="mpc_plotter",
        name="tt02_mpc_plotter",
        parameters=[
            sim_time_arg,
            {"waypoints_x": waypoints_x},
            {"waypoints_y": waypoints_y},
            {"scan_angle_offset_deg": LaunchConfiguration("plotter_scan_angle_offset_deg")},
            {
                "scan_mirror": ParameterValue(
                    LaunchConfiguration("plotter_scan_mirror"), value_type=bool
                )
            },
            {
                "scan_reverse": ParameterValue(
                    LaunchConfiguration("plotter_scan_reverse"), value_type=bool
                )
            },
            {"lidar_offset_x": LaunchConfiguration("plotter_lidar_offset_x")},
            {"lidar_offset_y": LaunchConfiguration("plotter_lidar_offset_y")},
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("run_plotter")),
    )

    return LaunchDescription([
        webots_host_arg,
        webots_port_arg,
        webots_robot_arg,
        run_mpc_arg,
        run_plotter_arg,
        plotter_scan_angle_offset_deg_arg,
        plotter_scan_mirror_arg,
        plotter_scan_reverse_arg,
        plotter_lidar_offset_x_arg,
        plotter_lidar_offset_y_arg,
        set_webots_url,
        driver_node,
        mpc_node,
        plotter_node,
    ])
