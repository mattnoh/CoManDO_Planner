from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    drone_name_arg = DeclareLaunchArgument(
        'drone_name',
        default_value='cf_1',
        description='Name of the Crazyflie drone'
    )
    
    world_frame_arg = DeclareLaunchArgument(
        'world_frame',
        default_value='world',
        description='World/map frame name'
    )
    
    control_freq_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='100.0',
        description='MPC control loop frequency in Hz'
    )
        
    # Planner node
    planner_node = Node(
        package='comando_planner',
        executable='comando_planner',
        name='comando_planner',
        output='screen',
        parameters=[{
            'drone_name': LaunchConfiguration('drone_name'),
            'world_frame': LaunchConfiguration('world_frame'),
            'control_frequency': LaunchConfiguration('control_frequency'),

        }],
        # Uncomment for debug logging:
        # arguments=['--ros-args', '--log-level', 'debug']
    )
    
    return LaunchDescription([
        drone_name_arg,
        world_frame_arg,
        control_freq_arg,
        planner_node,
    ])