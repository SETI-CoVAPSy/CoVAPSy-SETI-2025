from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_time_arg = {"use_sim_time": True}

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
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("run_mpc")),
    )

    return LaunchDescription([
        webots_host_arg,
        webots_port_arg,
        webots_robot_arg,
        run_mpc_arg,
        set_webots_url,
        driver_node,
        mpc_node,
    ])
