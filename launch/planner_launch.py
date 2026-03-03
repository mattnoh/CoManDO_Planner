from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    CoManDO Planner launch file.
    Parameters:
    - drone_name:     Name of the drone         (default: cf_1)
    - enable_logging: Enable CSV logging         (default: true)
    - ocp_type:       OCP formulation            (default: hover, options: hover, landing)
    - solver_rate:    MPC solver frequency in Hz  (default: 10)
    - platform:       Hardware platform           (default: crazyflie, options: crazyflie, px4)
    - solver:         Solver backend              (default: alipddp, options: alipddp, acados)
    - hover_target_x: Target hover x position    (default: 0.0)
    - hover_target_y: Target hover y position    (default: 0.0)
    - hover_target_z: Target hover z position    (default: 1.0)
    """
    
    # Declare launch arguments
    drone_name_arg = DeclareLaunchArgument(
        'drone_name',
        default_value='cf_1',
        description='Name of the drone'
    )
    
    enable_logging_arg = DeclareLaunchArgument(
        'enable_logging',
        default_value='true',
        description='Enable logging to CSV files'
    )
    
    ocp_type_arg = DeclareLaunchArgument(
        'ocp_type',
        default_value='hover',
        description='Type of OCP to use (hover, landing)'
    )
    
    solver_rate_arg = DeclareLaunchArgument(
        'solver_rate',
        default_value='10',
        description='MPC solver frequency in Hz'
    )

    platform_arg = DeclareLaunchArgument(
        'platform',
        default_value='crazyflie',
        description='Hardware platform (crazyflie, px4)'
    )

    solver_arg = DeclareLaunchArgument(
        'solver',
        default_value='alipddp',
        description='Solver backend (alipddp, acados)'
    )

    hover_target_x_arg = DeclareLaunchArgument(
        'hover_target_x',
        default_value='0.0',
        description='Target hover x position (m)'
    )

    hover_target_y_arg = DeclareLaunchArgument(
        'hover_target_y',
        default_value='0.0',
        description='Target hover y position (m)'
    )

    hover_target_z_arg = DeclareLaunchArgument(
        'hover_target_z',
        default_value='1.0',
        description='Target hover z position (m)'
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
            'solver_rate': LaunchConfiguration('solver_rate'),
            'platform': LaunchConfiguration('platform'),
            'solver': LaunchConfiguration('solver'),
            'hover_target_x': LaunchConfiguration('hover_target_x'),
            'hover_target_y': LaunchConfiguration('hover_target_y'),
            'hover_target_z': LaunchConfiguration('hover_target_z'),
        }]
    )
    
    return LaunchDescription([
        drone_name_arg,
        enable_logging_arg,
        ocp_type_arg,
        solver_rate_arg,
        platform_arg,
        solver_arg,
        hover_target_x_arg,
        hover_target_y_arg,
        hover_target_z_arg,
        planner_node,
    ])