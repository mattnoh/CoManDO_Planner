// ocp_hover_body.hpp
#pragma once

#include <Eigen/Dense>
#include <memory>

#include "ocp/ocp_hover.hpp"

namespace HoverBodyOCP {

// Reuse baseline hover constants to keep solver behavior predictable.
const int HORIZON = HoverOCP::HORIZON;
const double DT = HoverOCP::DT;
const double MASS = HoverOCP::MASS;
const int DEFAULT_N_REPLAY = HoverOCP::DEFAULT_N_REPLAY;
const double DEFAULT_MASS_KG = HoverOCP::DEFAULT_MASS_KG;

// cmd_vel_legacy thrust uses the system-id empirical polynomial (crazyflie-system-id).
// OCP mass = 0.0282 kg. Hover fz = 0.0282*9.81 ≈ 0.277 N → pwm ≈ 32560 u16.
// Limit FMAX to 0.40 N for headroom (pwm ≈ 42800 u16 < 60000 cap).
const double FMAX = 0.40;

inline Param getSolverParams() {
    return HoverOCP::getSolverParams();
}

// Body-hover test OCP:
//   - hold current position in the planner input frame
//   - drive velocity/angular-rate to zero
//   - keep neutral terminal quaternion target for mild attitude regularization
// This keeps hover thrust active so cmd_vel_legacy execution is observable.
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& terminal_state)
{
    // Use the planner-provided terminal_state for position (hover_target_x/y/z),
    // but override velocity/attitude/rates to zero for a clean hover reference.
    Eigen::VectorXd body_terminal = (terminal_state.size() >= 13)
        ? terminal_state
        : current_state;
    if (body_terminal.size() < 13) {
        body_terminal = Eigen::VectorXd::Zero(13);
    }

    body_terminal.segment<3>(3).setZero();   // desired v = 0
    body_terminal.segment<3>(10).setZero();  // desired omega = 0
    body_terminal(6) = 1.0;
    body_terminal(7) = 0.0;
    body_terminal(8) = 0.0;
    body_terminal(9) = 0.0;

    return HoverOCP::create(current_state, body_terminal, FMAX);
}

} // namespace HoverBodyOCP
