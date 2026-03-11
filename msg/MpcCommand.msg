# MpcCommand.msg
# ─────────────────────────────────────────────────────────────────────────────
# Published by comando_planner_node on /mpc/command.
# Subscribed to by every platform bridge (crazyflie_bridge, px4_bridge, ...).
#
# The bridge uses this message to:
#   1. Replay the full state trajectory at ocp_dt Hz
#   2. Apply the first control (fz, mx, my, mz) for direct torque platforms
#
# replay_start_idx already accounts for solve latency:
#   replay_start_idx = round(solve_duration_sec / ocp_dt)
#   The bridge starts replaying from this index, not from 0.
# ─────────────────────────────────────────────────────────────────────────────

std_msgs/Header header

bool   success
float64 solve_time_ms

# OCP timing — bridge uses ocp_dt to set its replay timer period
float64 ocp_dt
int32   horizon          # N  (number of steps; states = N+1, controls = N)
int32   nx               # state dimension  (13)
int32   nu               # control dimension (4: fz, Mx, My, Mz)
int32   n_shift

# Latency-compensated start index for replay
# Bridges start at index replay_start_idx, not 0
int32   replay_start_idx

# Full trajectories — flattened row-major
# state_trajectory  has (horizon+1)*nx  elements
# control_trajectory has  horizon*nu     elements
float64[] state_trajectory
float64[] control_trajectory

# First control u0 = control_trajectory[0] — extracted for convenience
# so bridges that only need the immediate command don't have to unpack the array
float64 fz    # collective thrust  (N)
float64 mx    # body moment x      (Nm, inertia-scaled)
float64 my    # body moment y      (Nm, inertia-scaled)
float64 mz    # body moment z      (Nm, inertia-scaled)
