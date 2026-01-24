from launch import LaunchDescription
from launch_ros.actions import Node

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

    # 2. RL Environment Node
    rl_env_node = Node(
        package='tt02_driver',
        executable='rl1_env',
        name='rl_environment',
        parameters=[sim_time_arg],
        output='screen'
    )

    # 3. RL Agent Node (TD3)
    rl_agent_node = Node(
        package='tt02_driver',
        executable='rl1_agent',
        name='rl_agent',
        parameters=[sim_time_arg],
        output='screen'
    )

    # 4. Static TF Publisher (base_link -> RpLidarA2)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.2', '0', '0.08', '0', '0', '0', 'base_link', 'RpLidarA2'],
        parameters=[sim_time_arg]
    )

    return LaunchDescription([
        driver_node,
        rl_env_node,
        rl_agent_node,
        static_tf_node
    ])
