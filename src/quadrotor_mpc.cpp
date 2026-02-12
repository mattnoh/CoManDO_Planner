#include "quadrotor_mpc.hpp"
#include "ocp_hover.hpp"
#include "ocp_constrained_attitude.hpp"
#include <iostream>

using namespace std;

QuadrotorMPC::QuadrotorMPC(const Config& config) : config_(config) {
    solver_params_.reg1_min  = 1e-6;
    solver_params_.reg2_min  = 1.0;
    solver_params_.mu_mul    = 0.1;
    solver_params_.rho       = 20.0;
    solver_params_.rho_mul   = 9.0;
    solver_params_.tolerance = 1e-3;

    // Constrained attitude has SOC + terminal EQ constraints that need more
    // augmented Lagrangian iterations to converge than the unconstrained hover.
    // This is the primary tuning knob if landing still stops short.
    if (config_.ocp_type == "constrained_attitude") {
        solver_params_.max_iter = 30;
    } else {
        solver_params_.max_iter = 10;
    }
}

double QuadrotorMPC::getOcpDt() const {
    if (config_.ocp_type == "hover")                return HoverOCP::DT;
    if (config_.ocp_type == "constrained_attitude") return ConstrainedAttitudeOCP::DT;
    return config_.dt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shift the previous solution one step forward for warm-starting.
// The last entry is held (repeated) rather than left undefined.
// ─────────────────────────────────────────────────────────────────────────────
void QuadrotorMPC::shiftWarmStart() {
    if (prev_X_.size() < 2 || prev_U_.empty()) return;

    vector<Eigen::VectorXd> sx(prev_X_.size());
    for (size_t i = 0; i + 1 < prev_X_.size(); ++i) sx[i] = prev_X_[i + 1];
    sx.back() = prev_X_.back();

    vector<Eigen::VectorXd> su(prev_U_.size());
    for (size_t i = 0; i + 1 < prev_U_.size(); ++i) su[i] = prev_U_[i + 1];
    su.back() = prev_U_.back();

    prev_X_ = move(sx);
    prev_U_ = move(su);
}

// ─────────────────────────────────────────────────────────────────────────────
void QuadrotorMPC::setupProblem(const Eigen::VectorXd& current_state) {
    if (config_.ocp_type == "hover") {
        problem_ = HoverOCP::create(current_state, config_.terminal_state);
    }
    else if (config_.ocp_type == "constrained_attitude") {
        problem_ = ConstrainedAttitudeOCP::create(current_state);
    }
    else {
        cerr << "ERROR: Unknown OCP type: " << config_.ocp_type << "\n";
        throw runtime_error("Unknown OCP type");
    }

    // Apply warm-start if available
    if (has_prev_solution_) {
        shiftWarmStart();
        for (int i = 0; i < (int)prev_X_.size(); ++i)
            problem_->setInitialState(i, prev_X_[i]);
        for (int i = 0; i < (int)prev_U_.size(); ++i)
            problem_->setInitialControl(i, prev_U_[i]);
    }
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

        result.solve_time_ms  = chrono::duration<double, milli>(
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
        has_prev_solution_ = false;   // goal changed — discard warm-start
    }
}