# Hover Controller Design Document

## Overview

This document describes the need for a centralized hover controller and the implementation plan to refactor hover hold logic from `planner_node.cpp` into a dedicated header file.

---

## Problem Statement

### Current Issues

1. **Code Duplication**: Hover hold logic is implemented inline in `planner_node.cpp` with similar patterns in multiple places:
   - `publishPausedHoverHoldTick()` ~40 lines
   - `holdHoverAndPause()` ~30 lines
   - Open loop mode needs same hover behavior but currently has no implementation

2. **File Length**: `planner_node.cpp` is ~788 lines, making it difficult to navigate and maintain.

3. **Inconsistent Behavior**: 
   - MPC mode has working hover hold when paused
   - Open loop mode falls after trajectory completes (no hover hold)
   - Open loop mode doesn't respond to `command_paused_` state

4. **Future Maintenance**: Adding hover behavior to open loop would duplicate more code.

---

## Goals

1. Extract hover hold logic into a reusable header-only library
2. Ensure both MPC and open loop modes share identical hover behavior
3. Reduce `planner_node.cpp` line count by ~100-150 lines
4. Make hover hold behavior consistent across all modes and states

---

## Hover Behavior Specification

### States Requiring Hover Hold

| State | MPC | Open Loop | Current Behavior | Target Behavior |
|-------|-----|-----------|------------------|-----------------|
| Unconfigured (waiting for `ocp_launch.py`) | No timers | No timers | N/A | N/A |
| Configured but paused (`command_paused_=true`) | Hover at `paused_hover_state_` | Falls | **Hover at current position** |
| Running trajectory | Execute trajectory | Execute trajectory | OK | OK |
| Trajectory complete | Hover via `holdHoverAndPause` | Falls | **Hover at current position** |
| Stale trajectory (3x warnings) | Hover via `holdHoverAndPause` | N/A | **Same for both** |

### Hover State Construction

When entering hover hold, construct a safe fullstate hover command:

```
Position:     Current drone position (from state estimator)
Velocity:     Zero (3x1)
Attitude:     Identity quaternion [1, 0, 0, 0] (flat, level)
Angular rate: Zero (3x1)
Thrust:       mass_kg * 9.81 (hover thrust)
Moments:      Zero (3x1)
```

### Key Principle

**Hover at current drone position**, not at trajectory final position. This handles:
- Leftover velocity in final trajectory steps
- Position drift during execution
- Safe, stable hover regardless of how trajectory ended

---

## Architecture

### Current Structure

```
planner_node.cpp (788 lines)
├── publishPausedHoverHoldTick()   [MPC only]
├── holdHoverAndPause()            [MPC only]
├── solverLoop()                   [MPC only]
├── mpcReplayTick()                [MPC only]
├── openLoopStartupCheck()         [Open loop only - no hover]
└── openLoopReplayTick()           [Open loop only - no hover]
```

### Proposed Structure

```
hover_controller.hpp (NEW, ~100 lines)
├── makeHoverState()
├── makeHoverControl()
├── doHoverHoldTick()
└── enterHoverHold()

planner_node.cpp (~650 lines)
├── publishPausedHoverHoldTick()   [calls hover_controller]
├── holdHoverAndPause()            [calls hover_controller]
├── solverLoop()                   [MPC]
├── mpcReplayTick()                [MPC]
├── openLoopStartupCheck()         [Open loop - now with hover]
└── openLoopReplayTick()           [Open loop - now with hover]
```

---

## Implementation Plan

### Phase 1: Create `hover_controller.hpp`

Create header-only library with inline functions:

```cpp
namespace hover_controller {

// Construct safe hover state from current state
Eigen::VectorXd makeHoverState(const Eigen::VectorXd& current_state);

// Construct hover control (thrust = mg, moments = 0)
Eigen::VectorXd makeHoverControl(double mass_kg);

// Execute one hover hold tick
// Returns true if hover command was published
template<typename PublishFunc>
bool doHoverHoldTick(
    bool is_configured,
    double mass_kg,
    bool maintain_hover_hold,
    bool has_state,
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& paused_hover_state,
    PublishFunc&& publish_cmd);

// Enter hover hold state and publish initial hover command
template<typename PublishFunc, typename ResetFunc>
void enterHoverHold(
    const std::string& reason,
    bool is_configured,
    double mass_kg,
    bool has_state,
    const Eigen::VectorXd& current_state,
    std::atomic<bool>& command_paused,
    bool& maintain_hover_hold,
    Eigen::VectorXd& paused_hover_state,
    PublishFunc&& publish_cmd,
    ResetFunc&& reset_cmd);

} // namespace hover_controller
```

### Phase 2: Refactor `planner_node.cpp`

1. Include `hover_controller.hpp`
2. Replace `publishPausedHoverHoldTick()` body with `hover_controller::doHoverHoldTick()`
3. Replace `holdHoverAndPause()` body with `hover_controller::enterHoverHold()`
4. Remove inline hover logic from `planner_node.cpp`

### Phase 3: Fix Open Loop Hover

1. Add `command_paused_` check to `openLoopStartupCheck()`:
   ```cpp
   if (command_paused_.load()) {
       publishPausedHoverHoldTick();
       return;
   }
   ```

2. Add `command_paused_` check to `openLoopReplayTick()`:
   ```cpp
   if (command_paused_.load()) {
       publishPausedHoverHoldTick();
       return;
   }
   ```

3. Add trajectory completion hover:
   ```cpp
   if (ol_replay_step_ >= N) {
       holdHoverAndPause("[OpenLoop] Trajectory complete");
       return;
   }
   ```

### Phase 4: Testing

| Test Case | Expected Behavior |
|-----------|-------------------|
| Start planner unconfigured | Log warning, no timers, no crash |
| Configure via `ocp_launch.py` | Timers created, ready message |
| `command_seq` not incremented | Hover at current position |
| `command_seq` incremented | Execute trajectory |
| MPC trajectory complete | Hover at current position |
| Open loop trajectory complete | Hover at current position |
| Open loop paused mid-trajectory | Hover at current position |

---

## Function Signatures

### `makeHoverState()`

```cpp
inline Eigen::VectorXd makeHoverState(const Eigen::VectorXd& current_state) {
    Eigen::VectorXd x_hover = (current_state.size() >= 13) 
        ? current_state 
        : Eigen::VectorXd::Zero(13);
    
    // Zero velocity
    x_hover.segment(3, 3).setZero();
    
    // Flat attitude (identity quaternion)
    x_hover(6) = 1.0;
    x_hover.segment(7, 3).setZero();
    
    // Zero angular velocity
    x_hover.segment(10, 3).setZero();
    
    return x_hover;
}
```

### `makeHoverControl()`

```cpp
inline Eigen::VectorXd makeHoverControl(double mass_kg) {
    Eigen::VectorXd u_hover = Eigen::VectorXd::Zero(4);
    u_hover(0) = mass_kg * 9.81;  // Thrust = mg
    return u_hover;
}
```

### `doHoverHoldTick()`

```cpp
template<typename PublishFunc>
inline bool doHoverHoldTick(
    bool is_configured,
    double mass_kg,
    bool maintain_hover_hold,
    bool has_state,
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& paused_hover_state,
    PublishFunc&& publish_cmd)
{
    if (!is_configured || mass_kg <= 0.0) return false;
    if (!maintain_hover_hold) return false;
    if (!has_state) return false;
    
    Eigen::VectorXd x_hold = (paused_hover_state.size() >= 13)
        ? paused_hover_state
        : makeHoverState(current_state);
    
    if (x_hold.size() < 13) return false;
    
    Eigen::VectorXd u_hover = makeHoverControl(mass_kg);
    publish_cmd(x_hold, u_hover);
    return true;
}
```

### `enterHoverHold()`

```cpp
template<typename PublishFunc, typename ResetFunc>
inline void enterHoverHold(
    const std::string& reason,
    bool is_configured,
    double mass_kg,
    bool has_state,
    const Eigen::VectorXd& current_state,
    std::atomic<bool>& command_paused,
    bool& maintain_hover_hold,
    Eigen::VectorXd& paused_hover_state,
    PublishFunc&& publish_cmd,
    ResetFunc&& reset_cmd)
{
    if (!is_configured || mass_kg <= 0.0) return;
    
    Eigen::VectorXd x_hover = has_state 
        ? makeHoverState(current_state) 
        : makeHoverState(Eigen::VectorXd::Zero(13));
    
    Eigen::VectorXd u_hover = makeHoverControl(mass_kg);
    publish_cmd(x_hover, u_hover);
    
    paused_hover_state = x_hover;
    maintain_hover_hold = true;
    command_paused.store(true);
    reset_cmd();
    
    // Log the reason for entering hover hold
    // (Caller should log with appropriate context)
}
```

---

## Integration with `planner_node.cpp`

### Before Refactoring

```cpp
void PlannerNode::publishPausedHoverHoldTick() {
    if (!is_configured_ || mass_kg_ <= 0.0) return;
    if (!maintain_hover_hold_) return;
    if (!hasState()) return;
    
    const Eigen::VectorXd x_now = getCurrentState();
    Eigen::VectorXd x_hold = (paused_hover_state_.size() >= 13)
        ? paused_hover_state_ : x_now;
    
    if (x_hold.size() < 13 || x_now.size() < 13) return;
    
    x_hold.segment(3, 3).setZero();
    x_hold.segment(10, 3).setZero();
    
    Eigen::VectorXd u_hover = Eigen::VectorXd::Zero(4);
    u_hover(0) = mass_kg_ * 9.81;
    publishCommand(x_hold, u_hover);
    
    // ... logging ...
}
```

### After Refactoring

```cpp
void PlannerNode::publishPausedHoverHoldTick() {
    hover_controller::doHoverHoldTick(
        is_configured_,
        mass_kg_,
        maintain_hover_hold_,
        hasState(),
        getCurrentState(),
        paused_hover_state_,
        [this](const auto& x, const auto& u) { publishCommand(x, u); }
    );
    
    // Logging handled separately or in doHoverHoldTick
}
```

---

## Open Loop Mode Changes

### Current `openLoopReplayTick()`

```cpp
void openLoopReplayTick() {
    const int N = static_cast<int>(ol_ref_X_.size()) - 1;
    const int step = std::min(ol_replay_step_, N);
    const Eigen::VectorXd& x_cmd = ol_ref_X_[step];
    const Eigen::VectorXd u_cmd = ...;
    
    publishCommand(x_cmd, u_cmd);
    
    if (ol_replay_step_ < N) {
        ++ol_replay_step_;
    } else if (!ol_done_logged_) {
        RCLCPP_INFO(this->get_logger(), "[OpenLoop] Done. Holding.");
        ol_done_logged_ = true;
    }
}
```

### Modified `openLoopReplayTick()`

```cpp
void openLoopReplayTick() {
    if (command_paused_.load()) {
        publishPausedHoverHoldTick();
        return;
    }
    
    const int N = static_cast<int>(ol_ref_X_.size()) - 1;
    
    if (ol_replay_step_ >= N) {
        holdHoverAndPause("[OpenLoop] Trajectory complete");
        return;
    }
    
    const int step = ol_replay_step_;
    const Eigen::VectorXd& x_cmd = ol_ref_X_[step];
    const Eigen::VectorXd u_cmd = ...;
    
    publishCommand(x_cmd, u_cmd);
    ++ol_replay_step_;
}
```

---

## Timeline

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | Create `hover_controller.hpp` | Low |
| 2 | Refactor `planner_node.cpp` | Medium |
| 3 | Fix open loop hover | Low |
| 4 | Testing | Medium |

**Total estimated effort:** ~2-3 hours

---

## Benefits

1. **Code Reuse**: Single source of truth for hover behavior
2. **Testability**: Hover functions can be unit tested independently
3. **Maintainability**: Changes to hover logic in one place
4. **Consistency**: Both modes use identical hover behavior
5. **Reduced Complexity**: `planner_node.cpp` becomes ~150 lines shorter

---

## Risks

1. **Template overhead**: Header-only templates may increase compile time slightly
2. **Refactoring bugs**: Must ensure existing MPC hover behavior is preserved
3. **Integration testing**: Need to verify both modes after refactoring

---

## References

- `planner_node.cpp`: Lines 211-280 (current hover implementation)
- `planner_logging.hpp`: Logging infrastructure
- `state_monitor.hpp`: State access patterns
