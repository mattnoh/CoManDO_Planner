#include "alipddp/alipddp.h"
#include "cost/error_quadratic_terminal_cost.h"
#include "cost/quadratic_stage_cost.h"
#include "dynamics/quad_flat_dynamics.h"
#include "optimal_control_problem.h"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

int main() {
    using Scalar = double;
    using Dyn = QuadPolyDirect<Scalar, 3, 0>;

    constexpr int kHorizon = 8;
    constexpr Scalar kDt = 0.25;

    auto dyn = std::make_shared<Dyn>(Dyn::TimeMode::Fixed, kDt);
    dyn->buildAB();

    OptimalControlProblem<Scalar> ocp(kHorizon);

    Eigen::VectorXd x0 = Dyn::makeHoverState(0.0, 0.0, 0.5);
    Eigen::VectorXd x_goal = Dyn::makeHoverState(1.0, 0.0, 0.5);

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(Dyn::NX_BASE, Dyn::NX_BASE);
    Eigen::MatrixXd R = 1.0e-3 * Eigen::MatrixXd::Identity(Dyn::NU_BASE, Dyn::NU_BASE);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(Dyn::NX_BASE, Dyn::NX_BASE);
    Qf.diagonal() << 400.0, 40.0, 2.0,
                     400.0, 40.0, 2.0,
                     400.0, 40.0, 2.0;

    auto stage_cost = std::make_shared<QuadraticStageCost<Scalar>>(Q, R);
    auto terminal_cost = std::make_shared<ErrorQuadraticTerminalCost<Scalar>>(Qf, x_goal);

    std::vector<Eigen::VectorXd> uwarm(
        kHorizon, Eigen::VectorXd::Zero(Dyn::NU_BASE));

    for (int k = 0; k < kHorizon; ++k) {
        ocp.setStageDynamics(k, dyn);
        ocp.setStageCost(k, stage_cost);
        ocp.setInitialControl(k, uwarm[k]);
        ocp.setInitialState(k, x0);
    }
    ocp.setInitialState(kHorizon, x0);
    ocp.setTerminalCost(terminal_cost);

    ALIPDDP<Scalar> solver(ocp);
    Param param;
    param.max_iter = 120;
    param.max_inner_iter = 40;
    param.tolerance = 1.0e-5;
    param.mu_min = 1.0e-6;
    solver.init(param);
    solver.solve();

    const auto X = solver.getResX();
    if (X.size() != static_cast<size_t>(kHorizon + 1)) {
        std::cerr << "Unexpected trajectory length: " << X.size() << "\n";
        return 1;
    }

    const double pos_error = (X.back().segment(0, 3) - x_goal.segment(0, 3)).norm();
    const bool converged = std::isfinite(solver.getError()) &&
                           solver.getError() < 1.0e-3 &&
                           pos_error < 5.0e-2;

    if (!converged) {
        std::cerr << "Standalone poly OCP failed. "
                  << "error=" << solver.getError()
                  << " pos_error=" << pos_error
                  << " reason=" << solver.getTerminationReason() << "\n";
        return 1;
    }

    std::cout << "Standalone poly OCP converged. "
              << "error=" << solver.getError()
              << " pos_error=" << pos_error
              << " reason=" << solver.getTerminationReason() << "\n";
    return 0;
}
