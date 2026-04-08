# include/

C++ header files for the CoManDO planner.

## Key subfolders / files
- `ocp/` – OCP (optimal control problem) formulations, e.g., `ocp_stateswitch.hpp`, `ocp_hover.hpp`, `ocp_landing.hpp`.
- `target/` – Target definitions such as `circular_target.hpp` and `target_accel_buffer.hpp`.
- `platform/` – Platform‑specific abstractions (`crazyflie.hpp`, `px4.hpp`).
- `planner_logging.hpp` – Logging utilities for the planner.
- `state_monitor.hpp` – Runtime state monitoring helpers.
- `quadrotor_mpc.hpp` – MPC interface definitions.
- `hover_controller.hpp` – Hover controller API.
- `trajectory_replayer.hpp` – Tools for replaying recorded trajectories.

These headers expose the core data structures and APIs used throughout the planner codebase.
