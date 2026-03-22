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
        return config_.dt;
    }
}

void QuadrotorMPC::shiftWarmStart(int n_shift) {
    if (prev_U_.empty()) return;
    const int Nu = (int)prev_U_.size();
    const int Nx = (int)prev_X_.size();
    n_shift = std::max(1, std::min(n_shift, Nu - 1));

    // Shift U
    std::vector<Eigen::VectorXd> su(Nu);
    for (int i = 0; i < Nu; ++i)
        su[i] = prev_U_[std::min(i + n_shift, Nu - 1)];
    prev_U_ = std::move(su);

    // Shift X — same tail-hold logic
    if (Nx > 0) {
        std::vector<Eigen::VectorXd> sx(Nx);
        for (int i = 0; i < Nx; ++i)
            sx[i] = prev_X_[std::min(i + n_shift, Nx - 1)];
        prev_X_ = std::move(sx);
    }
}

void QuadrotorMPC::setupProblem(const Eigen::VectorXd& current_state) {
    try {
        // CRITICAL: destroy the old solver BEFORE replacing the old problem.
        //
        // In every working single-file test (quad_cf_rh.cpp etc.) the solver
        // and problem are stack-allocated in the same scope, so C++ LIFO rules
        // guarantee the solver is destroyed FIRST (declared last), then the
        // problem is destroyed. Our shared_ptr members produce the REVERSE order:
        //
        //   problem_ = new_ocp  → old OCP ref-count drops to 0 → old OCP freed
        //   solver_  = new_slv  → old solver destructs AFTER old OCP already gone
        //
        // If ALIPDDP holds any raw pointer or reference into the OCP's stage
        // objects (dynamics, costs, constraints), its destructor accesses freed
        // memory — undefined behaviour that silently corrupts the heap and
        // poisons the newly constructed ALIPDDP on every subsequent solve.
        //
        // Resetting solver_ here forces the correct order:
        //   old solver freed (old OCP still alive)
        //   → old OCP freed
        //   → new OCP created
        solver_.reset();

        problem_ = OCPRegistry::create(
            config_.ocp_type, current_state, config_.terminal_state,
            prev_U_, prev_X_);
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        throw;
    }
}

QuadrotorMPC::Result QuadrotorMPC::solve(const Eigen::VectorXd& current_state)
{
    Result result;
    result.success = false;
    auto t0 = chrono::high_resolution_clock::now();

    try {
        // Pass the full measured state to the OCP — do NOT zero omega.
        // All working test cases (quad_cf_rh.cpp, quad_cf_rh_testing.cpp)
        // pass x_meas directly. Zeroing omega makes x0 inconsistent with
        // prev_U_ which was solved for a state that had real angular rates,
        // causing divergence on every warm-started solve after the first.
        if (!prev_U_.empty()) {
            shiftWarmStart(config_.n_shift);
        }

        setupProblem(current_state);

        solver_ = make_shared<ALIPDDP<double>>(*problem_);
        solver_->init(solver_params_);
        solver_->solve();

        auto X_result = solver_->getResX();
        auto U_result = solver_->getResU();

        result.solve_time_ms   = chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
        result.solve_timestamp = chrono::steady_clock::now();
        result.solve_iters     = static_cast<int>(solver_->getAllCost().size());
        last_solve_ms_         = result.solve_time_ms;

        if (X_result.size() > 1) {
            result.next_state         = X_result[1];
            result.state_trajectory   = X_result;
            result.control_trajectory = U_result;
            result.success            = true;
            prev_X_                   = X_result;
            prev_U_                   = U_result;
            std::cout << "[MPC] solve " << result.solve_time_ms
                      << "ms  iters=" << result.solve_iters
                      << "  n_shift=" << config_.n_shift << "\n";
        } else {
            std::cerr << "ERROR: Empty trajectory\n";
            last_solve_ms_ = 0.0;
        }

    } catch (const exception& e) {
        std::cerr << "ERROR in solve: " << e.what() << "\n";
        last_solve_ms_ = 0.0;
    }

    return result;
}

void QuadrotorMPC::setTerminalState(const Eigen::VectorXd& terminal) {
    if (terminal.size() == 13) {
        config_.terminal_state = terminal;
        solver_.reset();
        prev_U_.clear();
        prev_X_.clear();
        last_solve_ms_ = 0.0;
    }
}