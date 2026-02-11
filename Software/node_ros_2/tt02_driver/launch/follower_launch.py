from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    sim_time_arg = {'use_sim_time': True}

    # 1. Driver Node
    driver_node = Node(
        package='tt02_driver',
        executable='driver',
        name='tt02_driver',
        parameters=[{'target': 'simulation'}, sim_time_arg],
        output='screen'
    )

    # 2. Pilot Node (Auto-pilot)
    pilot_node = Node(
        package='tt02_driver',
        executable='pilot_alt',
        name='auto_pilot',
        parameters=[sim_time_arg],
        output='screen'
    )

    return LaunchDescription([
        driver_node,
        pilot_node,
    ])
