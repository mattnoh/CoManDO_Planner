#include "quadrotor_mpc.hpp"
#include "ocp_registry.hpp"
#include <iostream>

using namespace std;

QuadrotorMPC::QuadrotorMPC(const Config& config) : config_(config) {
    try {
        solver_params_ = OCPRegistry::getSolverParams(config_.ocp_type);
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        throw;
    }
}

double QuadrotorMPC::getOcpDt() const {
    try {
        return OCPRegistry::getDT(config_.ocp_type);
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return config_.dt;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void QuadrotorMPC::shiftWarmStart() {
    if (prev_X_.size() < 2 || prev_U_.empty()) return;
    const int Nx = (int)prev_X_.size();
    const int Nu = (int)prev_U_.size();
    const int n_shift = std::max(1, std::min(config_.n_shift, Nu - 1));
    {
        vector<Eigen::VectorXd> sx(Nx);
        for (int i = 0; i < Nx; ++i)
            sx[i] = prev_X_[std::min(i + n_shift, Nx - 1)];
        prev_X_ = move(sx);
    }
    {
        vector<Eigen::VectorXd> su(Nu);
        for (int i = 0; i < Nu; ++i)
            su[i] = prev_U_[std::min(i + n_shift, Nu - 1)];
        prev_U_ = move(su);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void QuadrotorMPC::setupProblem(const Eigen::VectorXd& current_state)
{
    if (!problem_ || need_problem_rebuild_) {
        try {
            problem_ = OCPRegistry::create(
                config_.ocp_type, current_state, config_.terminal_state, prev_U_);
        } catch (const std::runtime_error& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            throw;
        }
        need_problem_rebuild_ = false;
        solver_.reset();
        return;
    }
    problem_->setInitialState(0, current_state);
}

// ─────────────────────────────────────────────────────────────────────────────
// γ-blend: pulls warm-start away from constraint boundary toward hover.
//
// Motivation (Zhang et al. 2023, §III-B + Theorem 1):
//   The previous optimal solution sits ON the constraint boundary. After
//   shifting, prev_U_[0] was designed for a state at vz = -0.13 m/s.
//   Injecting it when actual vz ≈ 0 puts ALIPDDP near the boundary of the
//   new feasible region, causing search-direction blocking and divergence.
//
//   The paper's fix: blend the warm-start toward a well-centered cold-start
//   point:  u_warm = γ * u_shifted + (1-γ) * u_cold
//   where γ → 0 when state mismatch is large (vel_err large) and γ → 1 when
//   the drone is on-trajectory (vel_err small).
//
//   The γ formula below is heuristic but consistent with the paper's intent:
//   "selected γ should let the WSP have relatively small residuals and let
//   the point not be close to the boundary."
//
//   γ = max(0, 1 - vel_err / VEL_ERR_SCALE)
//     = 1.0  when vel_err = 0   → pure warm-start (no blend needed)
//     = 0.5  when vel_err = 0.15 m/s  → 50/50 mix
//     = 0.0  when vel_err ≥ 0.3 m/s  → pure hover cold-start
//
// VEL_ERR_SCALE = 0.3 m/s chosen from the jerk constraint:
//   jerk limit is |Δvz| ≤ 0.133 m/s per step. Two steps of Mellinger
//   lag gives ~0.26 m/s max expected vel_err during spin-up → scale at 0.3.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double VEL_ERR_SCALE = 0.3;   // m/s — vel_err at which γ → 0
static constexpr double CRAZYFLIE_MASS = 0.027; // kg

// ─────────────────────────────────────────────────────────────────────────────
QuadrotorMPC::Result QuadrotorMPC::solve(const Eigen::VectorXd& current_state)
{
    Result result;
    result.success = false;
    auto t0 = chrono::high_resolution_clock::now();

    try {
        // Zero angular velocity before passing to OCP.
        // Mellinger recomputes ω internally from attitude PD — its reactive
        // corrections are not predicted by the OCP dynamics and feeding them
        // back causes a state mismatch between x0_ocp and prev_X_[1].
        Eigen::VectorXd x0_ocp = current_state;
        x0_ocp.segment(10, 3).setZero();

        setupProblem(x0_ocp);

        if (!solver_) {
            // ── First solve: cold start ───────────────────────────────────────
            solver_ = make_shared<ALIPDDP<double>>(*problem_);
            solver_->init(solver_params_);
            

        } else {
            // ── Subsequent solves ─────────────────────────────────────────────
            std::cout << "x0_actual:  " << current_state.transpose() << "\n";
            std::cout << "prev_X_[1]: " << prev_X_[1].transpose() << "\n";
            const double pos_err  = (current_state.head(3)      - prev_X_[1].head(3)).norm();
            const double vel_err  = (current_state.segment(3,3)  - prev_X_[1].segment(3,3)).norm();
            const double quat_dot = std::abs(current_state.segment(6,4).dot(prev_X_[1].segment(6,4)));
            const double att_err  = 2.0 * std::acos(std::min(1.0, quat_dot)) * 180.0/M_PI;
            std::cout << "pos_err=" << pos_err
                    << " vel_err=" << vel_err
                    << " att_err_deg=" << att_err << "\n";

            shiftWarmStart();
            setupProblem(x0_ocp);

            // Recreate solver with the new problem — the old solver holds a copy
            // of the previous problem's initial state and won't see the new x0.
            solver_ = make_shared<ALIPDDP<double>>(*problem_);
            solver_->init(solver_params_);
            std::vector<Eigen::VectorXd> u_hover(prev_U_.size());
            for (auto& u : u_hover) {
                u = Eigen::VectorXd::Zero(4);
                u(0) = CRAZYFLIE_MASS * 9.81;
            }
            solver_->init(solver_params_);
            solver_->warmStart(x0_ocp, u_hover);
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
            cerr << "ERROR: Empty trajectory from solver\n";
            has_prev_solution_ = false;
        }

    } catch (const exception& e) {
        cerr << "ERROR in solve: " << e.what() << "\n";
        has_prev_solution_ = false;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
void QuadrotorMPC::setTerminalState(const Eigen::VectorXd& terminal) {
    if (terminal.size() == 13) {
        config_.terminal_state = terminal;
        has_prev_solution_    = false;
        need_problem_rebuild_ = true;
    }
}