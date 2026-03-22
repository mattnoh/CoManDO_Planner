/// @file ocp_registry.hpp
/// @brief Centralized OCP lookup and factory — reduces duplication and eases adding new OCPs

#pragma once

#include "ocp_hover.hpp"
#include "ocp_landing.hpp"
// future: #include "ocp_tracking.hpp"
// future: #include "ocp_avoid.hpp"

#include <string>
#include <stdexcept>
#include <memory>
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

namespace OCPRegistry {

// ── Per-OCP DT lookup ────────────────────────────────────────────────────────
inline double getDT(const std::string& ocp_type) {
    if (ocp_type == "hover")   return HoverOCP::DT;
    if (ocp_type == "landing") return LandingOCP::DT;
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}

// ── Per-OCP solver params lookup ──────────────────────────────────────────────
inline Param getSolverParams(const std::string& ocp_type) {
    Param p;
    if (ocp_type == "hover") {
        p.reg1_min  = HoverOCP::SOLVER_REG1_MIN;
        p.reg2_min  = HoverOCP::SOLVER_REG2_MIN;
        p.mu_mul    = HoverOCP::SOLVER_MU_MUL;
        p.rho       = HoverOCP::SOLVER_RHO;
        p.rho_mul   = HoverOCP::SOLVER_RHO_MUL;
        p.tolerance = HoverOCP::SOLVER_TOLERANCE;
        p.max_iter  = HoverOCP::SOLVER_MAX_ITER;
        p.is_quaternion_in_state = false;  // all working tests set this false
    } else if (ocp_type == "landing") {
        p.reg1_min  = LandingOCP::SOLVER_REG1_MIN;
        p.reg2_min  = LandingOCP::SOLVER_REG2_MIN;
        p.mu_mul    = LandingOCP::SOLVER_MU_MUL;
        p.rho       = LandingOCP::SOLVER_RHO;
        p.rho_mul   = LandingOCP::SOLVER_RHO_MUL;
        p.tolerance = LandingOCP::SOLVER_TOLERANCE;
        p.max_iter  = LandingOCP::SOLVER_MAX_ITER;
        p.rhoT      = LandingOCP::SOLVER_RHOT;
        p.is_quaternion_in_state = false;  // all working tests set this false
    } else {
        throw std::runtime_error("Unknown OCP type: " + ocp_type);
    }
    return p;
}

// ── Factory ───────────────────────────────────────────────────────────────────
// prev_U: shifted control warm-start from previous solve (empty on cold start).
// prev_X: shifted state trajectory from previous solve (empty on cold start).
//   Both are shifted by n_shift in QuadrotorMPC before being passed here.
//   LandingOCP::create uses prev_U to roll out the initial state guess and
//   prev_X for the trajectory-consistency stage cost reference.
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const std::string& ocp_type,
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& terminal_state = Eigen::VectorXd(),
    const std::vector<Eigen::VectorXd>& prev_U = {},
    const std::vector<Eigen::VectorXd>& prev_X = {})
{
    if (ocp_type == "hover")
        return HoverOCP::create(current_state, terminal_state);
    if (ocp_type == "landing")
        // NOTE: LandingOCP::create signature is (current, terminal, prev_U, prev_X).
        // Pass them in the correct order — prev_U first, prev_X second.
        return LandingOCP::create(current_state, terminal_state, prev_U, prev_X);
    throw std::runtime_error("Unknown OCP type: " + ocp_type);
}

} // namespace OCPRegistry