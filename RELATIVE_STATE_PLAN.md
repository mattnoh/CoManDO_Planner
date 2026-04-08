# Relative State Handling — CoManDO_planner

## One-Switch Design

Use one planner parameter to control where subtraction happens:

- `drone_state_is_relative: false`
  - Planner input state is absolute.
  - Registry applies transform before OCP creation:
    - `p_rel = p_drone - p_target`
    - `v_rel = v_drone - v_target`
- `drone_state_is_relative: true`
  - Planner input state is already relative.
  - Registry passes state through unchanged.

This means subtraction behavior is centralized in one place (planner/registry path), controlled by one parameter.

## OCP Factory Contract

Factories always consume `x0_rel`.

No factory-level subtraction is performed.

Affected factories:
- `stateswitch`
- `tracking_circle`
- `tracking_circle_target`

## Target Acceleration Rule

`a_tgt` is still applied in dynamics when target snapshot is valid.

Fallback behavior:
- valid target snapshot: `a_tgt = target.acceleration`
- invalid target snapshot: `a_tgt = [0, 0, 0]`

## Parameters Added

`planner_launch.py` now exposes:
- `drone_state_is_relative` (default `false`)
- `drone_odom_topic` (default empty)

When `drone_odom_topic` is non-empty (Crazyflie path), planner subscribes full state from that odometry topic directly.

Example for external relative state stream:
- `/drone/relative_odometry`

## Direct Relative-State Test (Stateswitch)

### A. Baseline (internal subtraction)

1. Start planner with default absolute state input:

```bash
ros2 launch comando_planner planner_launch.py \
  drone_state_is_relative:=false
```

2. Trigger stateswitch:

```bash
ros2 launch comando_planner ocp_launch.py \
  ocp_type:=stateswitch mode:=mpc
```

### B. Direct-relative input test

1. Start planner configured for direct-relative state input:

```bash
ros2 launch comando_planner planner_launch.py \
  drone_state_is_relative:=true \
  drone_odom_topic:=/drone/relative_odometry
```

2. Ensure an external node publishes `nav_msgs/msg/Odometry` on `/drone/relative_odometry` with:
- pose.position = relative position
- twist.linear = relative velocity
- pose.orientation and twist.angular = drone attitude and rates

3. Trigger stateswitch:

```bash
ros2 launch comando_planner ocp_launch.py \
  ocp_type:=stateswitch mode:=mpc
```

## Expected Equivalence Check

Compare A vs B runs under same scenario:
- solve success rate
- constraint error
- trajectory shape
- terminal relative error

If these match closely, direct-relative subscription path is correct.

## Notes

- `ocp_type:=stateswithc` is a typo. Use `ocp_type:=stateswitch`.
- `tracking_circle` and `tracking_circle_target` also follow the same `x0_rel` contract.
