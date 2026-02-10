from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Simplified launch file matching the example structure.
    Parameters:
    - drone_name: Name of the Crazyflie (default: cf_1)
    - enable_logging: Enable CSV logging (default: true)
    - ocp_type: Type of OCP to use (default: hover, options: hover, constrained_attitude)
    - control_rate: Control loop frequency in Hz (default: 50)
    - solver_rate: MPC solver frequency in Hz (default: 10)
    """
    
    # Declare launch arguments
    drone_name_arg = DeclareLaunchArgument(
        'drone_name',
        default_value='cf_1',
        description='Name of the Crazyflie drone'
    )
    
    enable_logging_arg = DeclareLaunchArgument(
        'enable_logging',
        default_value='true',
        description='Enable logging to CSV files'
    )
    
    ocp_type_arg = DeclareLaunchArgument(
        'ocp_type',
        default_value='hover',
        description='Type of OCP to use (hover, constrained_attitude)'
    )
    
    control_rate_arg = DeclareLaunchArgument(
        'control_rate',
        default_value='50',
        description='Control loop frequency in Hz'
    )
    
    solver_rate_arg = DeclareLaunchArgument(
        'solver_rate',
        default_value='10',
        description='MPC solver frequency in Hz'
    )
        
    # Planner node
    planner_node = Node(
        package='comando_planner',
        executable='comando_planner',
        name='comando_planner',
        output='screen',
        parameters=[{
            'drone_name': LaunchConfiguration('drone_name'),
            'enable_logging': LaunchConfiguration('enable_logging'),
            'ocp_type': LaunchConfiguration('ocp_type'),
            'control_rate': LaunchConfiguration('control_rate'),
            'solver_rate': LaunchConfiguration('solver_rate'),
        }]
    )
    
    return LaunchDescription([
        drone_name_arg,
        enable_logging_arg,
        ocp_type_arg,
        control_rate_arg,
        solver_rate_arg,
        planner_node,
    ])