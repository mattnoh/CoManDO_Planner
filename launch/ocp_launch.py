from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration


def _set_param(node, name, value):
    return ExecuteProcess(
        cmd=['ros2', 'param', 'set', node, name, value],
        output='screen'
    )


def generate_launch_description():
    """
    OCP trigger launcher.

    This launch file does NOT start the planner node.
    It sends an OCP/profile to a running planner and triggers command_seq.

    Terminal A:
    ros2 launch comando_planner planner_launch.py

    Terminal B:
    ros2 launch comando_planner ocp_launch.py \
        ocp_type:=landing mode:=open_loop n_replay:=5 command_seq:=1
    """

    args = [
        DeclareLaunchArgument('node_name', default_value='/comando_planner'),
        DeclareLaunchArgument('ocp_type', default_value='landing'),
        DeclareLaunchArgument('mode', default_value='mpc'),
        DeclareLaunchArgument('n_replay', default_value='5'),
        DeclareLaunchArgument('hover_target_x', default_value='0.0'),
        DeclareLaunchArgument('hover_target_y', default_value='0.0'),
        DeclareLaunchArgument('hover_target_z', default_value='0.0'),
        DeclareLaunchArgument('open_loop_abort_on_divergence', default_value='false'),
        DeclareLaunchArgument('open_loop_abort_max_z_error_m', default_value='0.50'),
        DeclareLaunchArgument('open_loop_abort_max_vz_error_mps', default_value='1.00'),
        DeclareLaunchArgument('command_seq', default_value='1'),
    ]

    node_name = LaunchConfiguration('node_name')

    set_params = [
        _set_param(node_name, 'ocp_type', LaunchConfiguration('ocp_type')),
        _set_param(node_name, 'mode', LaunchConfiguration('mode')),
        _set_param(node_name, 'n_replay', LaunchConfiguration('n_replay')),
        _set_param(node_name, 'hover_target_x', LaunchConfiguration('hover_target_x')),
        _set_param(node_name, 'hover_target_y', LaunchConfiguration('hover_target_y')),
        _set_param(node_name, 'hover_target_z', LaunchConfiguration('hover_target_z')),
        _set_param(node_name, 'open_loop_abort_on_divergence', LaunchConfiguration('open_loop_abort_on_divergence')),
        _set_param(node_name, 'open_loop_abort_max_z_error_m', LaunchConfiguration('open_loop_abort_max_z_error_m')),
        _set_param(node_name, 'open_loop_abort_max_vz_error_mps', LaunchConfiguration('open_loop_abort_max_vz_error_mps')),
    ]

    trigger = _set_param(node_name, 'command_seq', LaunchConfiguration('command_seq'))

    actions = []
    for i, act in enumerate(set_params):
        actions.append(TimerAction(period=0.15 * i, actions=[act]))
    actions.append(TimerAction(period=0.15 * len(set_params) + 0.2, actions=[trigger]))

    return LaunchDescription(args + actions)
