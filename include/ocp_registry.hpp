/// @file ocp_registry.hpp
/// @brief Centralized OCP lookup and factory — reduces duplication and eases adding new OCPs

#pragma once

#include "ocp/ocp_hover.hpp"
#include "ocp/ocp_landing.hpp"
#include "ocp/ocp_stateswitch.hpp"

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
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}

inline int getDefaultNReplay(const std::string& ocp_type) {
    if (ocp_type == "hover") return HoverOCP::DEFAULT_N_REPLAY;
    if (ocp_type == "landing") return LandingOCP::DEFAULT_N_REPLAY;
    if (ocp_type == "stateswitch") return StateswitchOCP::DEFAULT_N_REPLAY;
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}

inline double getDefaultMassKg(const std::string& ocp_type) {
    if (ocp_type == "hover") return HoverOCP::DEFAULT_MASS_KG;
    if (ocp_type == "landing") return LandingOCP::DEFAULT_MASS_KG;
    if (ocp_type == "stateswitch") return StateswitchOCP::DEFAULT_MASS_KG;
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}

inline Param getSolverParams(const std::string& ocp_type) {
    if (ocp_type == "hover") return HoverOCP::getSolverParams();
    if (ocp_type == "landing") return LandingOCP::getSolverParams();
    if (ocp_type == "stateswitch") return StateswitchOCP::getSolverParams();
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}

inline std::shared_ptr<OptimalControlProblem<double>> create(
    const std::string& ocp_type,
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& terminal_state = Eigen::VectorXd(),
    const std::vector<Eigen::VectorXd>& prev_U = {},
    const std::vector<Eigen::VectorXd>& prev_X = {},
    const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero(),
    const std::vector<Eigen::MatrixXd>& prev_K = {}) {
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
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}

}  // namespace OCPRegistry
