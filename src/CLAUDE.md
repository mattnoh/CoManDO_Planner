# src/

C++ source files for the CoManDO planner.

## Main components
- `planner_node.cpp` – Implements ROS node that orchestrates planning, subscribes to sensor data, and publishes control commands.
- `quadrotor_mpc.cpp` – MPC controller implementation for quadrotor dynamics.
- `target_publisher.cpp` – Publishes target trajectories (e.g., circular or dynamic targets) to the planner.

These files compile into the planner executable defined in the workspace’s `CMakeLists.txt`.
