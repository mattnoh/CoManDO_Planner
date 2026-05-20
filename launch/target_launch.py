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
        DeclareLaunchArgument('rigid_body_name', default_value='stmini'),
        DeclareLaunchArgument('drone_odom_topic', default_value='/cf_1/odom'),
        DeclareLaunchArgument('drone_pose_topic', default_value='/cf_1/pose'),
        DeclareLaunchArgument('body_relative_input_angular_unit', default_value='deg_s'),
        DeclareLaunchArgument('debug_body_relative_trace', default_value='false'),
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
            'rigid_body_name': LaunchConfiguration('rigid_body_name'),
            'drone_odom_topic': LaunchConfiguration('drone_odom_topic'),
            'drone_pose_topic': LaunchConfiguration('drone_pose_topic'),
            'body_relative_input_angular_unit': LaunchConfiguration('body_relative_input_angular_unit'),
            'debug_body_relative_trace': LaunchConfiguration('debug_body_relative_trace'),
        }],
    )

    return LaunchDescription(args + [node])
