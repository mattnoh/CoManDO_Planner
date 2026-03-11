/// @file ocp_base.hpp
/// @brief Shared includes and documentation anchor for all CoManDO OCPs.
///
/// Every OCP in include/ocp/ should include this header first.
/// It pulls in the external solver interfaces (optimal_control_problem.h,
/// Quad6DOF dynamics) so each OCP file stays focused on its own cost / constraint
/// definitions rather than boilerplate.
///
/// ── Adding a new OCP ─────────────────────────────────────────────────────────
///   1. Create  include/ocp/ocp_<name>.hpp
///   2. Define a namespace <Name>OCP { ... } with:
///        - Physical constants  (HORIZON, DT, MASS, ...)
///        - Solver params       (SOLVER_REG1_MIN, SOLVER_MAX_ITER, ...)
///        - Cost classes        (StageCost<Scalar>, TerminalCost<Scalar>)
///        - Constraint classes  (as needed)
///        - Factory function    create(current_state, terminal_state, ...)
///   3. Add two lines in include/solver/ocp_registry.hpp:
///        getDT()          — if (ocp_type == "<name>") return <Name>OCP::DT;
///        getSolverParams() — populate Param from <Name>OCP constants
///        create()          — if (ocp_type == "<name>") return <Name>OCP::create(...)
///   That's it. No other file needs to change.
/// ─────────────────────────────────────────────────────────────────────────────

#pragma once

// External solver framework (ALIPDDP problem API)
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

// Eigen
#include <Eigen/Dense>

// Standard
#include <cmath>
#include <memory>
#include <vector>