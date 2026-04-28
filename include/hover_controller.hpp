#pragma once

#include <Eigen/Dense>
#include <atomic>
#include <string>

namespace hover_controller {

// Construct safe hover state from current state
inline Eigen::VectorXd makeHoverState(
    const Eigen::VectorXd& current_state,
    int state_dim,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> custom_make_hover_state) 
{
    if (custom_make_hover_state) {
        return custom_make_hover_state(current_state);
    }
    return Eigen::VectorXd::Zero(state_dim);
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
    int state_dim,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> custom_make_hover_state,
    bool maintain_hover_hold,
    bool has_state,
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& paused_hover_state,
    PublishFunc&& publish_cmd)
{
    if (!is_configured || mass_kg <= 0.0) return false;
    if (!maintain_hover_hold) return false;
    if (!has_state) return false;
    
    Eigen::VectorXd x_hold = (paused_hover_state.size() >= state_dim)
        ? paused_hover_state
        : makeHoverState(current_state, state_dim, custom_make_hover_state);
    
    if (x_hold.size() < state_dim) return false;
    
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
    int state_dim,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> custom_make_hover_state,
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
        ? makeHoverState(current_state, state_dim, custom_make_hover_state) 
        : makeHoverState(Eigen::VectorXd::Zero(state_dim), state_dim, custom_make_hover_state);
    
    Eigen::VectorXd u_hover = makeHoverControl(mass_kg);
    publish_cmd(x_hover, u_hover);
    
    paused_hover_state = x_hover;
    maintain_hover_hold = true;
    command_paused.store(true);
    reset_cmd();
    
    log_reason(reason.c_str());
}

} // namespace hover_controller
