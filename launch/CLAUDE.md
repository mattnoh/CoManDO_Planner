# launch/

Python launch helpers for starting the CoManDO planner ROS nodes.

## Scripts
- `planner_launch.py` – Launches the main planner node, loads the runtime configuration, and starts necessary ROS components.
- `ocp_launch.py` – Launches OCP‑specific processes (e.g., optimal‑control problem solvers) and configures related parameters.

Both scripts are used with `ros2 launch` to bring up the system in various modes.
