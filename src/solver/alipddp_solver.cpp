/// @file alipddp_solver.cpp
/// @brief ALIPDDP MPC solver implementation.
///
/// ── Key design invariant ─────────────────────────────────────────────────────
///   solver_ is created ONCE on the first (cold) solve and reused for every
///   subsequent warm solve. warmStart() carries over IPM dual variables (S, Y, e).
///
/// ── Why setupProblem is NOT called on warm start ──────────────────────────────
///   solver_ = make_shared<ALIPDDP<double>>(*problem_) copies problem_ by value.
///   After construction, problem_ and solver_'s internal problem are SEPARATE objects.
///   Calling problem_->setInitialState() on warm start modifies the disconnected
///   problem_ — it does nothing to solver_. The standalone never calls setInitialState
///   between warm solves either. solver_->warmStart(x0, u_warm) handles the initial
///   condition update internally. setupProblem is only called on cold start (or
///   when need_problem_rebuild_ is true).
///
/// ── sim_n_shift ──────────────────────────────────────────────────────────────
///   When config_.sim_n_shift > 0, shiftWarmStart() uses that value instead of
///   the timing-computed config_.n_shift.
///   Set sim_n_shift=1 in sim to match standalone N_SHIFT=1.
///   Launch planner: --ros-args -p sim_n_shift:=1

#include "solver/alipddp_solver.hpp"
#include "solver/ocp_registry.hpp"
#include "utils/logger.hpp"
#include <iostream>

using namespace std;

ALIPDDPSolver::ALIPDDPSolver(const Config& config) : config_(config) {
    try {
        solver_params_ = OCPRegistry::getSolverParams(config_.ocp_type);
    } catch (const std::runtime_error& e) {
        COMANDO_ERROR("ALIPDDPSolver", "%s", e.what());
        throw;
    }
}

double ALIPDDPSolver::getOcpDt() const {
    try {
        return OCPRegistry::getDT(config_.ocp_type);
    } catch (const std::runtime_error& e) {
        COMANDO_ERROR("ALIPDDPSolver", "%s", e.what());
        return config_.dt;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void ALIPDDPSolver::shiftWarmStart() {
    if (prev_X_.size() < 2 || prev_U_.empty()) return;
    const int Nx = (int)prev_X_.size();
    const int Nu = (int)prev_U_.size();

    // sim_n_shift=1 matches standalone N_SHIFT=1 exactly.
    // Hardware uses timing-computed n_shift (sim_n_shift=0).
    const int shift = (config_.sim_n_shift > 0)
                    ? config_.sim_n_shift
                    : std::max(1, std::min(config_.n_shift, Nu - 1));

    {
        vector<Eigen::VectorXd> sx(Nx);
        for (int i = 0; i < Nx; ++i)
            sx[i] = prev_X_[std::min(i + shift, Nx - 1)];
        prev_X_ = move(sx);
    }
    {
        vector<Eigen::VectorXd> su(Nu);
        for (int i = 0; i < Nu; ++i)
            su[i] = prev_U_[std::min(i + shift, Nu - 1)];
        prev_U_ = move(su);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void ALIPDDPSolver::setupProblem(const Eigen::VectorXd& current_state)
{
    // Always rebuild on first call or when terminal state changes.
    // For warm starts, this is NOT called — warmStart() handles the initial
    // condition update internally on the solver's own copy of the problem.
    if (!problem_ || need_problem_rebuild_) {
        try {
            problem_ = OCPRegistry::create(
                config_.ocp_type, current_state, config_.terminal_state, prev_U_);
        } catch (const std::runtime_error& e) {
            COMANDO_ERROR("ALIPDDPSolver", "%s", e.what());
            throw;
        }
        need_problem_rebuild_ = false;
        solver_.reset();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
ALIPDDPSolver::Result ALIPDDPSolver::solve(const Eigen::VectorXd& current_state)
{
    Result result;
    result.success = false;
    auto t0 = chrono::high_resolution_clock::now();

    try {
        const Eigen::VectorXd& x0_ocp = current_state;

        if (!solver_) {
            // ── Cold start ────────────────────────────────────────────────────
            // setupProblem builds problem_ and resets solver_.
            // After this, solver_ has its OWN copy of problem_.
            // Modifying problem_ after this point does NOT affect solver_.
            setupProblem(x0_ocp);
            solver_ = make_shared<ALIPDDP<double>>(*problem_);
            solver_->init(solver_params_);

            COMANDO_INFO("ALIPDDPSolver", "Cold start at x0=[%.3f,%.3f,%.3f]",
                x0_ocp(0), x0_ocp(1), x0_ocp(2));

        } else {
            // ── Warm start ────────────────────────────────────────────────────
            // Diagnostics: warm-start quality (same metric as dxws_pred in standalone)
            cout << "x0_actual:  " << current_state.transpose() << "\n";
            if (prev_X_.size() > 1) {
                cout << "prev_X_[1]: " << prev_X_[1].transpose() << "\n";
                const double pos_err  = (current_state.head(3)     - prev_X_[1].head(3)).norm();
                const double vel_err  = (current_state.segment(3,3) - prev_X_[1].segment(3,3)).norm();
                const double quat_dot = std::abs(current_state.segment(6,4).dot(prev_X_[1].segment(6,4)));
                const double att_err  = 2.0 * std::acos(std::min(1.0, quat_dot)) * 180.0 / M_PI;
                cout << "pos_err=" << pos_err
                     << " vel_err=" << vel_err
                     << " att_err_deg=" << att_err << "\n";
            }

            // Shift warm-start by sim_n_shift (sim) or n_shift (hardware).
            shiftWarmStart();

            // warmStart handles the initial condition update inside solver_.
            // Do NOT call setupProblem here — it would only modify the disconnected
            // problem_ copy and do nothing to solver_.
            solver_->warmStart(x0_ocp, prev_U_);
        }

        solver_->solve();

        auto X_result = solver_->getResX();
        auto U_result = solver_->getResU();

        result.solve_time_ms   = chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
        result.solve_timestamp = chrono::steady_clock::now();

        if (X_result.size() > 1) {
            result.next_state         = X_result[1];
            result.state_trajectory   = X_result;
            result.control_trajectory = U_result;
            result.success            = true;

            prev_X_            = X_result;
            prev_U_            = U_result;
            has_prev_solution_ = true;

            for (int i = 0; i < std::min(20, (int)X_result.size()); ++i)
                cout << "X[" << i << "]: " << X_result[i].transpose() << "\n";
        } else {
            COMANDO_ERROR("ALIPDDPSolver", "Empty trajectory from solver");
            has_prev_solution_ = false;
        }

    } catch (const exception& e) {
        COMANDO_ERROR("ALIPDDPSolver", "Exception in solve: %s", e.what());
        has_prev_solution_ = false;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
void ALIPDDPSolver::setTerminalState(const Eigen::VectorXd& terminal) {
    if (terminal.size() == 13) {
        config_.terminal_state = terminal;
        has_prev_solution_    = false;
        need_problem_rebuild_ = true;
        solver_.reset();
    }
}