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
    - ocp_type:        OCP formulation            (default: hover, options: hover, landing)
    - solver_rate:     MPC solver frequency in Hz (default: 1 → replan every 1s, replay 10 pts at 10Hz)
    - platform:        Hardware platform           (default: crazyflie, options: crazyflie, px4)
    - solver:          Solver backend              (default: alipddp, options: alipddp, acados)
    - mode:            Execution mode              (default: mpc, options: mpc, open_loop)
    - hover_target_x:  Landing target x           (default: 0.0)
    - hover_target_y:  Landing target y           (default: 0.0)
    - hover_target_z:  Landing target z           (default: 0.0 for landing, 1.0 for hover)
    """

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
        default_value='1',
        description=(
            'MPC solver frequency in Hz. '
            'With ocp_dt=0.1s: solver_rate=1 → n_shift=10, replays X[1..10] at 10Hz between solves. '
            'solver_rate=2 → n_shift=5, replays X[1..5]. '
            'solver_rate=10 → n_shift=1 (old behaviour, one point per solve).'
        )
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
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='mpc',
        description='Execution mode (mpc, open_loop)'
    )
    hover_target_x_arg = DeclareLaunchArgument(
        'hover_target_x',
        default_value='0.0',
        description='Target x position (m)'
    )
    hover_target_y_arg = DeclareLaunchArgument(
        'hover_target_y',
        default_value='0.0',
        description='Target y position (m)'
    )
    hover_target_z_arg = DeclareLaunchArgument(
        'hover_target_z',
        default_value='0.1',
        description='Target z position (m) — 0.0 for landing, 1.0 for hover'
    )

    planner_node = Node(
        package='comando_planner',
        executable='comando_planner',
        name='comando_planner',
        output='screen',
        parameters=[{
            'drone_name':      LaunchConfiguration('drone_name'),
            'enable_logging':  LaunchConfiguration('enable_logging'),
            'ocp_type':        LaunchConfiguration('ocp_type'),
            'solver_rate':     LaunchConfiguration('solver_rate'),
            'platform':        LaunchConfiguration('platform'),
            'solver':          LaunchConfiguration('solver'),
            'mode':            LaunchConfiguration('mode'),
            'hover_target_x':  LaunchConfiguration('hover_target_x'),
            'hover_target_y':  LaunchConfiguration('hover_target_y'),
            'hover_target_z':  LaunchConfiguration('hover_target_z'),
        }]
    )

    return LaunchDescription([
        drone_name_arg,
        enable_logging_arg,
        ocp_type_arg,
        solver_rate_arg,
        platform_arg,
        solver_arg,
        mode_arg,
        hover_target_x_arg,
        hover_target_y_arg,
        hover_target_z_arg,
        planner_node,
    ])