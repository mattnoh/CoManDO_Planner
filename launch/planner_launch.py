from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessIO
from launch.substitutions import LaunchConfiguration, PythonExpression
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
        DeclareLaunchArgument('max_first_solve_age_sec', default_value='1.00'),
        DeclareLaunchArgument('max_relative_position_norm', default_value='10.0'),
        DeclareLaunchArgument('max_relative_vertical_abs', default_value='5.0'),
        DeclareLaunchArgument('max_relative_velocity_norm', default_value='8.0'),

        DeclareLaunchArgument('body_relative_odom_topic', default_value='/drone/body_relative_odom'),

        DeclareLaunchArgument('target_odom_topic', default_value='/target/odom'),
        DeclareLaunchArgument('target_accel_topic', default_value='/target/accel'),
        DeclareLaunchArgument('target_predicted_accel_topic', default_value='/target/predicted_accel'),

        DeclareLaunchArgument('enable_terminal_freeze', default_value='true'),
        DeclareLaunchArgument('terminal_freeze_enter_pos', default_value='0.20'),
        DeclareLaunchArgument('terminal_freeze_enter_vel', default_value='0.10'),
        DeclareLaunchArgument('terminal_freeze_require_vel', default_value='false'),
        DeclareLaunchArgument('terminal_freeze_exit_pos', default_value='0.20'),

        DeclareLaunchArgument('record_bag', default_value='true'),
        DeclareLaunchArgument('bag_output', default_value=''),
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
            'max_first_solve_age_sec': LaunchConfiguration('max_first_solve_age_sec'),
            'max_relative_position_norm': LaunchConfiguration('max_relative_position_norm'),
            'max_relative_vertical_abs': LaunchConfiguration('max_relative_vertical_abs'),
            'max_relative_velocity_norm': LaunchConfiguration('max_relative_velocity_norm'),

            'body_relative_odom_topic': LaunchConfiguration('body_relative_odom_topic'),

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

    drone_topic = lambda suffix: PythonExpression([
        "'/' + '", LaunchConfiguration('drone_name'), "' + '", suffix, "'"
    ])

    def make_rosbag(output_path):
        return ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('record_bag')),
            cmd=[
                'ros2', 'bag', 'record',
                '-o', output_path,
                '/tf',
                '/tf_static',
                drone_topic('/planned_trajectory'),
                drone_topic('/planner_debug_markers'),
                '/target/odom',
                '/target/accel',
                '/target/true_odom',
                '/target/true_accel',
                '/target/predicted_accel',
                '/drone/body_relative_odom',
                '/drone/target_frame_odom',
                drone_topic('/pose'),
                drone_topic('/odom'),
                drone_topic('/cmd_hover'),
                drone_topic('/cmd_full_state'),
                '/mavros/local_position/odom',
                '/mavros/setpoint_raw/attitude',
                '/mavros/setpoint_raw/local',
            ],
            output='screen',
        )

    bag_state = {'started': False, 'log_folder': '', 'profile_applied': False}

    def start_bag_after_profile_applied(event):
        if bag_state['started']:
            return []
        text = event.text.decode(errors='ignore') if isinstance(event.text, bytes) else str(event.text)
        if 'Profile applied. Set command_seq to start this command.' in text:
            bag_state['profile_applied'] = True
        marker = 'Logging to:'
        if marker in text:
            log_folder = text.split(marker, 1)[1].splitlines()[0].strip()
            if log_folder:
                bag_state['log_folder'] = log_folder
        if not bag_state['profile_applied'] or not bag_state['log_folder']:
            return []
        bag_state['started'] = True
        default_output = (
            f"{bag_state['log_folder'].rstrip('/')}/bags/comando_debug"
            if bag_state['log_folder'] else 'bags/comando_debug'
        )
        output_path = PythonExpression([
            "'", default_output, "' if '", LaunchConfiguration('bag_output'),
            "' == '' else '", LaunchConfiguration('bag_output'), "'"
        ])
        return [
            LogInfo(
                condition=IfCondition(LaunchConfiguration('record_bag')),
                msg='Planner profile applied; starting rosbag recording before command execution.'
            ),
            make_rosbag(output_path),
        ]

    delayed_rosbag_start = RegisterEventHandler(
        OnProcessIO(
            target_action=node,
            on_stdout=start_bag_after_profile_applied,
            on_stderr=start_bag_after_profile_applied,
        )
    )

    return LaunchDescription(args + [node, delayed_rosbag_start])
