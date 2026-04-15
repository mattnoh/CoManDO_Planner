from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Bare planner launcher - infrastructure only.

    Starts /comando_planner with subscriptions/publishers active but unconfigured.
    Use ocp_launch.py to push an OCP profile and trigger command_seq.

    OCP-related params (ocp_type, mode, n_replay, hover_target) are NOT set here.
    They must be configured via ocp_launch.py before execution.
    """

    args = [
        DeclareLaunchArgument('drone_name', default_value='cf_1'),
        DeclareLaunchArgument('platform', default_value='crazyflie'),
        DeclareLaunchArgument('solver', default_value='alipddp'),
        DeclareLaunchArgument('enable_logging', default_value='true'),
        DeclareLaunchArgument('open_loop_abort_on_divergence', default_value='false'),
        DeclareLaunchArgument('open_loop_abort_max_z_error_m', default_value='0.50'),
        DeclareLaunchArgument('open_loop_abort_max_vz_error_mps', default_value='1.00'),

        DeclareLaunchArgument('drone_state_is_relative', default_value='false'),
        DeclareLaunchArgument('drone_odom_topic', default_value=''),

        DeclareLaunchArgument('target_odom_topic', default_value='/target/odom'),
        DeclareLaunchArgument('target_accel_topic', default_value='/target/accel'),
        DeclareLaunchArgument('target_predicted_accel_topic', default_value='/target/predicted_accel'),

        DeclareLaunchArgument('enable_terminal_freeze', default_value='true'),
        DeclareLaunchArgument('terminal_freeze_enter_pos', default_value='0.20'),
        DeclareLaunchArgument('terminal_freeze_enter_vel', default_value='0.10'),
        DeclareLaunchArgument('terminal_freeze_require_vel', default_value='false'),
        DeclareLaunchArgument('terminal_freeze_exit_pos', default_value='0.20'),
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
            'open_loop_abort_on_divergence': LaunchConfiguration('open_loop_abort_on_divergence'),
            'open_loop_abort_max_z_error_m': LaunchConfiguration('open_loop_abort_max_z_error_m'),
            'open_loop_abort_max_vz_error_mps': LaunchConfiguration('open_loop_abort_max_vz_error_mps'),

            'drone_state_is_relative': LaunchConfiguration('drone_state_is_relative'),
            'drone_odom_topic': LaunchConfiguration('drone_odom_topic'),

            'target_odom_topic': LaunchConfiguration('target_odom_topic'),
            'target_accel_topic': LaunchConfiguration('target_accel_topic'),
            'target_predicted_accel_topic': LaunchConfiguration('target_predicted_accel_topic'),

            'enable_terminal_freeze': LaunchConfiguration('enable_terminal_freeze'),
            'terminal_freeze_enter_pos': LaunchConfiguration('terminal_freeze_enter_pos'),
            'terminal_freeze_enter_vel': LaunchConfiguration('terminal_freeze_enter_vel'),
            'terminal_freeze_require_vel': LaunchConfiguration('terminal_freeze_require_vel'),
            'terminal_freeze_exit_pos': LaunchConfiguration('terminal_freeze_exit_pos'),

            'start_paused': True,
            'command_seq': 0,
        }]
    )

    return LaunchDescription(args + [node])
