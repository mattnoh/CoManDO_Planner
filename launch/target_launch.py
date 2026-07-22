from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('target_mode', default_value='gazebo_circle'),
        DeclareLaunchArgument('planning_frame', default_value='world'),
        DeclareLaunchArgument('target_yaw_rate', default_value='0.3'),
        DeclareLaunchArgument('publish_hz', default_value='100.0'),
        DeclareLaunchArgument('frame_id', default_value='world'),
        DeclareLaunchArgument('trace_csv_path', default_value=''),
        DeclareLaunchArgument('trace_loop', default_value='false'),
        DeclareLaunchArgument('trace_hold_last', default_value='true'),
        DeclareLaunchArgument('trace_time_scale', default_value='1.0'),
        DeclareLaunchArgument('trace_shifted_world', default_value='false'),
        DeclareLaunchArgument('trace_offset_x', default_value='0.0'),
        DeclareLaunchArgument('trace_offset_y', default_value='0.0'),
        DeclareLaunchArgument('trace_offset_z', default_value='0.0'),
        DeclareLaunchArgument('rigid_body_name', default_value='stmini'),
        DeclareLaunchArgument('drone_odom_topic', default_value='/cf_1/odom'),
        DeclareLaunchArgument('drone_pose_topic', default_value='/cf_1/pose'),
        DeclareLaunchArgument('body_relative_input_angular_unit', default_value='deg_s'),
        DeclareLaunchArgument('debug_body_relative_trace', default_value='false'),
        # Simulated relative sensing (defaults preserve legacy behavior)
        DeclareLaunchArgument('target_source', default_value='ground_truth'),
        DeclareLaunchArgument('sim_relative_estimate', default_value='false'),
        DeclareLaunchArgument('pad_z', default_value='0.0'),
        DeclareLaunchArgument('relative_estimate_topic',
                              default_value='/drone/relative_target_estimate'),
        DeclareLaunchArgument('publish_ground_truth_debug', default_value='true'),
        DeclareLaunchArgument('drone_odom_twist_frame', default_value='body'),
    ]

    node = Node(
        package='comando_planner',
        executable='target_publisher',
        name='target_publisher',
        output='screen',
        parameters=[{
            'target_mode': LaunchConfiguration('target_mode'),
            'planning_frame': LaunchConfiguration('planning_frame'),
            'target_yaw_rate': LaunchConfiguration('target_yaw_rate'),
            'publish_hz': LaunchConfiguration('publish_hz'),
            'frame_id': LaunchConfiguration('frame_id'),
            'trace_csv_path': LaunchConfiguration('trace_csv_path'),
            'trace_loop': LaunchConfiguration('trace_loop'),
            'trace_hold_last': LaunchConfiguration('trace_hold_last'),
            'trace_time_scale': LaunchConfiguration('trace_time_scale'),
            'trace_shifted_world': LaunchConfiguration('trace_shifted_world'),
            'trace_offset_x': LaunchConfiguration('trace_offset_x'),
            'trace_offset_y': LaunchConfiguration('trace_offset_y'),
            'trace_offset_z': LaunchConfiguration('trace_offset_z'),
            'rigid_body_name': LaunchConfiguration('rigid_body_name'),
            'drone_odom_topic': LaunchConfiguration('drone_odom_topic'),
            'drone_pose_topic': LaunchConfiguration('drone_pose_topic'),
            'body_relative_input_angular_unit': LaunchConfiguration('body_relative_input_angular_unit'),
            'debug_body_relative_trace': LaunchConfiguration('debug_body_relative_trace'),
            'target_source': LaunchConfiguration('target_source'),
            'sim_relative_estimate': LaunchConfiguration('sim_relative_estimate'),
            'pad_z': LaunchConfiguration('pad_z'),
            'relative_estimate_topic': LaunchConfiguration('relative_estimate_topic'),
            'publish_ground_truth_debug': LaunchConfiguration('publish_ground_truth_debug'),
            'drone_odom_twist_frame': LaunchConfiguration('drone_odom_twist_frame'),
        }],
    )

    return LaunchDescription(args + [node])
