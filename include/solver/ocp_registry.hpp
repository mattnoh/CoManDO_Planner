/// @file ocp_registry.hpp
/// @brief Centralized OCP lookup and factory.
///
/// Adding a new OCP: three lines here + one new file in include/ocp/.
/// Nothing else in the codebase changes.

#pragma once

#include "ocp/ocp_hover.hpp"
#include "ocp/ocp_landing.hpp"
// future: #include "ocp/ocp_tracking.hpp"
// future: #include "ocp/ocp_avoid.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

namespace OCPRegistry {

// ── DT lookup ────────────────────────────────────────────────────────────────
inline double getDT(const std::string& ocp_type) {
    if (ocp_type == "hover")   return HoverOCP::DT;
    if (ocp_type == "landing") return LandingOCP::DT;
    throw std::runtime_error("OCPRegistry: unknown OCP type '" + ocp_type + "'");
}

// ── Solver params lookup ──────────────────────────────────────────────────────
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
    } else if (ocp_type == "landing") {
        p.reg1_min  = LandingOCP::SOLVER_REG1_MIN;
        p.reg2_min  = LandingOCP::SOLVER_REG2_MIN;
        p.mu_mul    = LandingOCP::SOLVER_MU_MUL;
        p.rho       = LandingOCP::SOLVER_RHO;
        p.rho_mul   = LandingOCP::SOLVER_RHO_MUL;
        p.tolerance = LandingOCP::SOLVER_TOLERANCE;
        p.max_iter  = LandingOCP::SOLVER_MAX_ITER;
        p.rhoT      = LandingOCP::SOLVER_RHOT;
    } else {
        throw std::runtime_error("OCPRegistry: unknown OCP type '" + ocp_type + "'");
    }
    return p;
}

// ── Factory ───────────────────────────────────────────────────────────────────
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
        return LandingOCP::create(current_state, terminal_state, prev_X, prev_U);
    throw std::runtime_error("OCPRegistry: unknown OCP type '" + ocp_type + "'");
}

} // namespace OCPRegistry