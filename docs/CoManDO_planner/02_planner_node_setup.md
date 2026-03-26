# Chapter 2: Planner Node Setup & Lifecycle

## 1. Multi-Threaded Architecture

To ensure high-frequency command streaming and low-latency state updates, the `comando_planner` node uses three independent **Callback Groups**:

1.  **Sensor/State Group (`sensor_cb_group_`)**:
    -   Handles `/pose`, `/odom`, and `/target` subscriptions.
    -   Writes to the thread-safe `StateMonitor`.
2.  **Solver Group (`solver_cb_group_`)**:
    -   Runs the `solverLoop()` at a 1ms polling rate.
    -   Executes the ALIPDDP solve when `n_replay` ticks have elapsed.
3.  **Replay Group (`replay_cb_group_`)**:
    -   Runs the `mpcReplayTick()` at precisely `ocp_dt` intervals.
    -   Samples the trajectory and publishes `FullState` commands.

This decoupling prevents the heavy MPC computation (which may take 10-50ms) from jittering the 50Hz command stream.

---

## 2. Solving Logic (The Registry System)

The node is OCP-agnostic. It loads a **Descriptor** from the `OCPRegistry` based on the `ocp_type` parameter.

### 2.1 Coordinate Transformations

The Registry allows for custom state pre-processing. For example:
-   **Absolute OCPs** (`hover`, `landing`): Use the measured state $x$ directly.
-   **Relative OCPs** (`stateswitch`): The Registry's `transform_state` callback subtracts the target position/velocity from the drone state *before* solving.

### 2.2 Post-Processing

Some OCPs (like `tracking_circle`) solve in a fixed relative frame for speed but require the output trajectory to be shifted back to absolute coordinates. This is handled by the `post_process_result` callback.

---

## 3. The `n_replay` Timing Model

The balance between "prediction horizon" and "control rate" is controlled by two variables:
-   **`ocp_dt`**: The time-step of the OCP (e.g., 50ms).
-   **`n_replay`**: How many steps to "replay" before re-solving.

**Example**: If `ocp_dt = 0.05s` and `n_replay = 4`, the planner solves every `0.05 * 4 = 0.2s` (5 Hz), but it sends commands every `0.05s` (20 Hz).

---

## 4. Node Lifecycle & Execution

### 4.1 Configuration Phase
The node starts in an **Unconfigured** state if `ocp_type` is not set. It will subscribe to topics but will not solve.

### 4.2 Priming & Execution
1.  **Set Parameters**: Configure `ocp_type`, `mode`, and `target`.
2.  **Trigger `command_seq`**: Incrementing the `command_seq` parameter tells the node to "Apply" the current profile and start execution.
3.  **State Wait**: The node waits until it has both `pose` and `odom` (and a fresh target if needed).
4.  **Solving**: Once primed, the `solverLoop` starts generating trajectories.

### 4.3 Terminal Freeze
When the drone is within `terminal_freeze_enter_pos` of the target, the solver pauses to save CPU. It resumes automatically if the drone drifts beyond `terminal_freeze_exit_pos` (hysteresis).

---

## 5. Summary of Parameters

### Core Infrastructure
| Parameter | Default | Description |
|-----------|---------|-------------|
| `drone_name` | `"cf_1"` | Target drone |
| `platform` | `"crazyflie"` | Hardware driver logic |
| `enable_logging`| `true` | CSV output toggle |

### Execution Control
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ocp_type` | `""` | e.g., `"hover"`, `"tracking_circle"` |
| `mode` | `"mpc"` | `"mpc"` or `"open_loop"` |
| `n_replay` | `4` | Solve interval in dt-steps |
| `command_seq` | `0` | Trigger for new commands |

---

[Next Chapter: How the MPC Works](03_mpc.md)