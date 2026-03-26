from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Solver-ready launcher.

    Starts /comando_planner with subscriptions/publishers active and solver paused.
    Use ocp_launch.py to push an OCP profile and trigger command_seq.
    """

    args = [
        DeclareLaunchArgument('drone_name', default_value='cf_1'),
        DeclareLaunchArgument('platform', default_value='crazyflie'),
        DeclareLaunchArgument('solver', default_value='alipddp'),
        DeclareLaunchArgument('enable_logging', default_value='true'),

        DeclareLaunchArgument('target_odom_topic', default_value='/target/odom'),
        DeclareLaunchArgument('target_accel_topic', default_value='/target/accel'),

        DeclareLaunchArgument('enable_terminal_freeze', default_value='true'),
        DeclareLaunchArgument('terminal_freeze_enter_pos', default_value='0.20'),
        DeclareLaunchArgument('terminal_freeze_enter_vel', default_value='0.10'),
        DeclareLaunchArgument('terminal_freeze_require_vel', default_value='false'),
        DeclareLaunchArgument('terminal_freeze_exit_pos', default_value='0.20'),

        # Baseline defaults used before an OCP is commanded.
        DeclareLaunchArgument('ocp_type', default_value='landing'),
        DeclareLaunchArgument('mode', default_value='mpc'),
        DeclareLaunchArgument('hover_target_x', default_value='0.0'),
        DeclareLaunchArgument('hover_target_y', default_value='0.0'),
        DeclareLaunchArgument('hover_target_z', default_value='1.0'),
        DeclareLaunchArgument('n_replay', default_value='4'),
        DeclareLaunchArgument('mass_kg', default_value='0.027'),
    ]

    node = Node(
        package='comando_planner',
        executable='comando_planner',
        name='comando_planner',
        output='screen',
        parameters=[{
            'drone_name': LaunchConfiguration('drone_name'),
            'platform': LaunchConfiguration('platform'),
            'solver': LaunchConfiguration('solver'),
            'enable_logging': LaunchConfiguration('enable_logging'),

            'target_odom_topic': LaunchConfiguration('target_odom_topic'),
            'target_accel_topic': LaunchConfiguration('target_accel_topic'),

            'enable_terminal_freeze': LaunchConfiguration('enable_terminal_freeze'),
            'terminal_freeze_enter_pos': LaunchConfiguration('terminal_freeze_enter_pos'),
            'terminal_freeze_enter_vel': LaunchConfiguration('terminal_freeze_enter_vel'),
            'terminal_freeze_require_vel': LaunchConfiguration('terminal_freeze_require_vel'),
            'terminal_freeze_exit_pos': LaunchConfiguration('terminal_freeze_exit_pos'),

            'ocp_type': LaunchConfiguration('ocp_type'),
            'mode': LaunchConfiguration('mode'),
            'hover_target_x': LaunchConfiguration('hover_target_x'),
            'hover_target_y': LaunchConfiguration('hover_target_y'),
            'hover_target_z': LaunchConfiguration('hover_target_z'),
            'n_replay': LaunchConfiguration('n_replay'),
            'mass_kg': LaunchConfiguration('mass_kg'),

            # Solver starts ready but paused.
            'start_paused': True,
            'command_seq': 0,
        }]
    )

    return LaunchDescription(args + [node])
