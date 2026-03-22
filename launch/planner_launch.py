from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    CoManDO Planner launch file.
    Parameters:
    - drone_name:      Name of the drone         (default: cf_1)
    - enable_logging:  Enable CSV logging         (default: true)
    - ocp_type:        OCP formulation            (default: landing)
    - n_replay:        Setpoints sent between solves (default: 4 → 200ms at ocp_dt=50ms)
                       Change to 10 for 500ms between solves.
                       n_replay * ocp_dt = solve interval.
    - platform:        Hardware platform           (default: crazyflie)
    - solver:          Solver backend              (default: alipddp)
    - mode:            Execution mode              (default: mpc)
    - hover_target_x:  Target x                   (default: 0.0)
    - hover_target_y:  Target y                   (default: 0.0)
    - hover_target_z:  Target z (0.0=land, 1.0=hover) (default: 0.0)
    """

    args = [
        DeclareLaunchArgument('drone_name',     default_value='cf_1'),
        DeclareLaunchArgument('enable_logging', default_value='true'),
        DeclareLaunchArgument('ocp_type',       default_value='landing'),
        DeclareLaunchArgument('n_replay',       default_value='4',
            description='Setpoints sent per solve cycle. '
                        'n_replay * ocp_dt_ms = solve interval in ms. '
                        'ocp_dt=50ms: n_replay=4→200ms, n_replay=10→500ms.'),
        DeclareLaunchArgument('platform',       default_value='crazyflie'),
        DeclareLaunchArgument('solver',         default_value='alipddp'),
        DeclareLaunchArgument('mode',           default_value='mpc'),
        DeclareLaunchArgument('hover_target_x', default_value='0.0'),
        DeclareLaunchArgument('hover_target_y', default_value='0.0'),
        DeclareLaunchArgument('hover_target_z', default_value='0.0'),
    ]

    planner_node = Node(
        package='comando_planner',
        executable='comando_planner',
        name='comando_planner',
        output='screen',
        parameters=[{
            'drone_name':      LaunchConfiguration('drone_name'),
            'enable_logging':  LaunchConfiguration('enable_logging'),
            'ocp_type':        LaunchConfiguration('ocp_type'),
            'n_replay':        LaunchConfiguration('n_replay'),
            'platform':        LaunchConfiguration('platform'),
            'solver':          LaunchConfiguration('solver'),
            'mode':            LaunchConfiguration('mode'),
            'hover_target_x':  LaunchConfiguration('hover_target_x'),
            'hover_target_y':  LaunchConfiguration('hover_target_y'),
            'hover_target_z':  LaunchConfiguration('hover_target_z'),
        }]
    )

    return LaunchDescription(args + [planner_node])