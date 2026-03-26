#pragma once

#include <Eigen/Dense>
#include <atomic>
#include <string>

namespace hover_controller {

// Construct safe hover state from current state
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

// Construct hover control (thrust = mg, moments = 0)
inline Eigen::VectorXd makeHoverControl(double mass_kg) {
    Eigen::VectorXd u_hover = Eigen::VectorXd::Zero(4);
    u_hover(0) = mass_kg * 9.81;  // Thrust = mg
    return u_hover;
}

// Execute one hover hold tick
// Returns true if hover command was published
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

// Enter hover hold state and publish initial hover command
template<typename PublishFunc, typename LoggerFunc, typename ResetFunc>
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
    LoggerFunc&& log_reason,
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
    
    log_reason(reason.c_str());
}

} // namespace hover_controller
