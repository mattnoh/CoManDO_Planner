#include "quadrotor_mpc.hpp"
#include "ocp_hover.hpp"
#include "ocp_landing.hpp"
#include <iostream>

using namespace std;

QuadrotorMPC::QuadrotorMPC(const Config& config) : config_(config) {
    solver_params_.reg1_min  = 1e-6;
    solver_params_.reg2_min  = 1.0;
    solver_params_.mu_mul    = 0.1;
    solver_params_.rho       = 20.0;
    solver_params_.rho_mul   = 9.0;
    solver_params_.tolerance = 1e-3;
    solver_params_.max_iter  = 200;
}

double QuadrotorMPC::getOcpDt() const {
    if (config_.ocp_type == "hover")   return HoverOCP::DT;
    if (config_.ocp_type == "landing") return LandingOCP::DT;
    return config_.dt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shift the previous solution one step forward for warm-starting.
//
// WHAT THE SOLVER ACTUALLY USES:
//   DDP only reads U as its initial guess. x[0] is the fixed measured state
//   and x[1..N] are always recomputed by the solver's own forward rollout —
//   they are overwritten on the very first forward sweep of every iteration.
//
// WHAT WE DO WITH prev_X_:
//   We shift it anyway and keep it in memory. It is NOT passed to the solver
//   right now. Kept intentionally for future use if we extend the solver backend
//   to accept an X seed (e.g. a custom initialiser).
//
// WHAT WE DO WITH prev_U_:
//   This IS the actual warm-start seed. We shift it forward one step so that
//   the previous optimal controls initialise the next solve window:
//     u_warm[k] ← u_prev[k+1]   for k = 0, ..., N-2
//     u_warm[N-1] ← u_prev[N-1]  (hold last control)
// ─────────────────────────────────────────────────────────────────────────────
void QuadrotorMPC::shiftWarmStart() {
    if (prev_X_.size() < 2 || prev_U_.empty()) return;

    // Shift X — NOT sent to the solver, kept for future use only.
    vector<Eigen::VectorXd> sx(prev_X_.size());
    for (size_t i = 0; i + 1 < prev_X_.size(); ++i) sx[i] = prev_X_[i + 1];
    sx.back() = prev_X_.back();
    prev_X_ = move(sx);

    // Shift U — this IS the DDP warm-start seed.
    vector<Eigen::VectorXd> su(prev_U_.size());
    for (size_t i = 0; i + 1 < prev_U_.size(); ++i) su[i] = prev_U_[i + 1];
    su.back() = prev_U_.back();
    prev_U_ = move(su);
}

// ─────────────────────────────────────────────────────────────────────────────
void QuadrotorMPC::setupProblem(const Eigen::VectorXd& current_state) {
    if (config_.ocp_type == "hover") {
        problem_ = HoverOCP::create(current_state, config_.terminal_state);
    }
    else if (config_.ocp_type == "landing") {
        problem_ = LandingOCP::create(current_state);
    }
    else {
        cerr << "ERROR: Unknown OCP type: " << config_.ocp_type << "\n";
        throw runtime_error("Unknown OCP type");
    }

    // ── x[0]: always the real measured state ─────────────────────────────────
    // DDP reads only x[0] from outside. x[1..N] are computed by the solver's
    // own forward rollout — setting them here would be immediately overwritten.
    problem_->setInitialState(0, current_state);

    if (has_prev_solution_) {
        // ── Warm start ────────────────────────────────────────────────────────
        // Shift U one step forward and pass it as the initial control sequence.
        // This overrides whatever create() set for U, which is what we want —
        // the previous optimal solution is a much better starting point than
        // the generic gravity-hover seed that create() provides.
        //
        // REMOVED: the old code also ran a manual forward rollout here:
        //   x_next = f(x, u)  →  setInitialState(i+1, x_next)
        // That was redundant — DDP overwrites x[1..N] in its forward sweep
        // before any backward pass, so those states had no effect on the solve.
        shiftWarmStart();

        for (int i = 0; i < (int)prev_U_.size(); ++i)
            problem_->setInitialControl(i, prev_U_[i]);

    }
    // ── Cold start ────────────────────────────────────────────────────────────
    // No else branch needed. Both HoverOCP::create() and LandingOCP::create()
    // already seed U with a gravity-compensating hover computed from the actual
    // current quaternion:
    //   f0 = q^{-1} * [0, 0, m*g]   (body-frame thrust to cancel gravity)
    //   u0 = [f0, 0, 0, 0]
    // This is correct and better than anything we could add here, so we leave it.
}

// ─────────────────────────────────────────────────────────────────────────────
QuadrotorMPC::Result QuadrotorMPC::solve(const Eigen::VectorXd& current_state) {
    Result result;
    result.success = false;
    auto t0 = chrono::high_resolution_clock::now();

    try {
        setupProblem(current_state);

        if (!problem_) {
            cerr << "ERROR: Problem not initialized\n";
            return result;
        }

        solver_.reset();
        solver_ = make_shared<ALIPDDP<double>>(*problem_);
        solver_->init(solver_params_);
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

            // Cache full solution for the next warm start.
            // prev_X_ is not read by the solver — stored for future use only.
            // prev_U_ is the actual warm-start seed passed to the next solve.
            prev_X_            = X_result;
            prev_U_            = U_result;
            has_prev_solution_ = true;

            for (int i = 0; i < min(4, (int)X_result.size()); ++i)
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
        has_prev_solution_ = false;
    }
}