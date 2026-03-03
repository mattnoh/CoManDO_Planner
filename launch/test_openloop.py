from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Open-Loop Divergence Test launch file.

    Run this INSTEAD of the normal planner when you suspect the receding-horizon
    MPC is diverging due to problems in the first-solve trajectory itself
    (bad dynamics model, wrong mass, frame mismatch, etc.).

    What it does:
      1. Waits for the first /odom message.
      2. Runs the MPC solver EXACTLY ONCE.
      3. Replays the resulting open-loop trajectory step-by-step at ocp_dt.
      4. Records commanded vs actual states into:
           ./logs/<drone>_open_loop_test_<timestamp>/
             planned_trajectory.csv      — the raw MPC solution
             commanded_vs_actual.csv     — side-by-side comparison per step

    Interpretation:
      actual ≈ commanded   →  first-solve is fine; debug the feedback loop
      actual diverges      →  fix the OCP dynamics / cost first

    Parameters (same defaults as planner_launch.py for easy swap):
    - drone_name:     Name of the drone         (default: cf_1)
    - enable_logging: Enable CSV logging         (default: true)
    - ocp_type:       OCP formulation            (default: hover)
    - platform:       Hardware platform           (default: crazyflie)
    - hover_target_x: Target x position          (default: 0.0)
    - hover_target_y: Target y position          (default: 0.0)
    - hover_target_z: Target z position          (default: 1.0)
    """

    drone_name_arg = DeclareLaunchArgument(
        'drone_name',
        default_value='cf_1',
        description='Name of the drone'
    )

    enable_logging_arg = DeclareLaunchArgument(
        'enable_logging',
        default_value='true',
        description='Enable CSV logging'
    )

    ocp_type_arg = DeclareLaunchArgument(
        'ocp_type',
        default_value='hover',
        description='OCP type (hover, landing)'
    )

    platform_arg = DeclareLaunchArgument(
        'platform',
        default_value='crazyflie',
        description='Hardware platform (crazyflie, px4)'
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
        default_value='1.0',
        description='Target z position (m)'
    )

    open_loop_test_node = Node(
        package='comando_planner',
        executable='open_loop_test',          # add to CMakeLists.txt
        name='open_loop_test_node',
        output='screen',
        parameters=[{
            'drone_name':     LaunchConfiguration('drone_name'),
            'enable_logging': LaunchConfiguration('enable_logging'),
            'ocp_type':       LaunchConfiguration('ocp_type'),
            'platform':       LaunchConfiguration('platform'),
            'hover_target_x': LaunchConfiguration('hover_target_x'),
            'hover_target_y': LaunchConfiguration('hover_target_y'),
            'hover_target_z': LaunchConfiguration('hover_target_z'),
        }]
    )

    return LaunchDescription([
        drone_name_arg,
        enable_logging_arg,
        ocp_type_arg,
        platform_arg,
        hover_target_x_arg,
        hover_target_y_arg,
        hover_target_z_arg,
        open_loop_test_node,
    ])