/// @file ocp_registry.hpp
/// @brief Centralized OCP lookup and factory — reduces duplication and eases adding new OCPs
///
/// BUG 3 FIX: Updated create() to accept live circle_target and t_abs parameters
/// for tracking_circle OCP, passing them from QuadrotorMPC::Config.

#pragma once


#include "ocp/ocp_hover.hpp"
#include "ocp/ocp_landing.hpp"
#include "ocp/ocp_stateswitch.hpp"
#include "ocp/ocp_tracking_circle.hpp"

#include <string>
#include <stdexcept>
#include <memory>

#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

namespace OCPRegistry {


inline double getDT(const std::string& ocp_type) {
    if (ocp_type == "hover") return HoverOCP::DT;
    if (ocp_type == "landing") return LandingOCP::DT;
    if (ocp_type == "stateswitch") return StateswitchOCP::TH_INIT;
    if (ocp_type == "tracking_circle") return TrackingCircleOCP::TH_INIT;
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}


inline int getDefaultNReplay(const std::string& ocp_type) {
    if (ocp_type == "hover") return HoverOCP::DEFAULT_N_REPLAY;
    if (ocp_type == "landing") return LandingOCP::DEFAULT_N_REPLAY;
    if (ocp_type == "stateswitch") return StateswitchOCP::DEFAULT_N_REPLAY;
    if (ocp_type == "tracking_circle") return TrackingCircleOCP::DEFAULT_N_REPLAY;
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}


inline double getDefaultMassKg(const std::string& ocp_type) {
    if (ocp_type == "hover") return HoverOCP::DEFAULT_MASS_KG;
    if (ocp_type == "landing") return LandingOCP::DEFAULT_MASS_KG;
    if (ocp_type == "stateswitch") return StateswitchOCP::DEFAULT_MASS_KG;
    if (ocp_type == "tracking_circle") return TrackingCircleOCP::MASS;
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}


inline Param getSolverParams(const std::string& ocp_type) {
    if (ocp_type == "hover") return HoverOCP::getSolverParams();
    if (ocp_type == "landing") return LandingOCP::getSolverParams();
    if (ocp_type == "stateswitch") return StateswitchOCP::getSolverParams();
    if (ocp_type == "tracking_circle") return TrackingCircleOCP::getSolverParams();
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}

/// @brief Creates an OCP instance based on type.
/// @param ocp_type One of: "hover", "landing", "stateswitch", "tracking_circle"
/// @param current_state Current robot state
/// @param terminal_state Terminal state constraint (for hover/landing)
/// @param prev_U Warm-start controls from previous solve
/// @param prev_X Warm-start states from previous solve
/// @param target_accel Target acceleration (for stateswitch)
/// @param prev_K Warm-start feedback gains from previous solve
/// @param circle_target CircularTarget parameters (for tracking_circle)
/// @param t_abs Absolute time at solve start (for tracking_circle)
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const std::string& ocp_type,
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& terminal_state = Eigen::VectorXd(),
    const std::vector<Eigen::VectorXd>& prev_U = {},
    const std::vector<Eigen::VectorXd>& prev_X = {},
    const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero(),
    const std::vector<Eigen::MatrixXd>& prev_K = {},
    const TrackingCircleOCP::CircularTarget& circle_target = TrackingCircleOCP::CircularTarget{},
    double t_abs = 0.0
) {
    if (ocp_type == "hover") {
        return HoverOCP::create(current_state, terminal_state);
    }
    if (ocp_type == "landing") {
        return LandingOCP::create(current_state, terminal_state, prev_U, prev_X);
    }
    if (ocp_type == "stateswitch") {
        return StateswitchOCP::create(current_state, terminal_state, prev_U, prev_X,
                                       target_accel, prev_K);
    }
    if (ocp_type == "tracking_circle") {
        // BUG 3 FIX: Use live circle_target and t_abs from QuadrotorMPC::Config
        // instead of hardcoded placeholder values.
        return TrackingCircleOCP::create(current_state, circle_target, t_abs,
                                          TrackingCircleOCP::TH_INIT,
                                          TrackingCircleOCP::THL,
                                          TrackingCircleOCP::THH,
                                          nullptr, prev_U);
    }
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}

} // namespace OCPRegistry
