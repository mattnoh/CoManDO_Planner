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
    n_shift = std::max(1, std::min(n_shift, Nu - 1));

    // Shift U only — not X.
    // The solver rolls out X internally from x0 during init().
    // Shifting a stale X overwrites that rollout and kills warm-start.
    std::vector<Eigen::VectorXd> su(Nu);
    for (int i = 0; i < Nu; ++i)
        su[i] = prev_U_[std::min(i + n_shift, Nu - 1)];
    prev_U_ = std::move(su);
}

void QuadrotorMPC::setupProblem(const Eigen::VectorXd& current_state) {
    try {
        problem_ = OCPRegistry::create(
            config_.ocp_type, current_state, config_.terminal_state,
            prev_U_, {});
        const int N = (int)prev_U_.size();
        for (int i = 0; i < N; ++i)
            problem_->setInitialControl(i, prev_U_[i]);
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
        Eigen::VectorXd x0_ocp = current_state;
        x0_ocp.segment(10, 3).setZero();

        if (!prev_U_.empty()) {
            // n_shift is fixed = config_.n_shift = n_replay_ (set by PlannerNode).
            // This is the number of replay ticks that fire between solves, so it
            // is constant regardless of actual solve wall-time.
            // Using last_solve_ms_ to compute this dynamically was wrong: when
            // solve time drifts from n_replay_*ocp_dt the warm-start U is shifted
            // by a different amount than the replay timer has actually advanced,
            // producing a misaligned (corrupted) warm-start every subsequent solve.
            shiftWarmStart(config_.n_shift);
        }

        // Rebuild problem with shifted warm-start.
        setupProblem(x0_ocp);

        // Recreate solver from the NEW problem every solve.
        // ALIPDDP copies the problem at construction, so this is required
        // to pick up the rebuilt OCP and updated initial controls.
        solver_ = make_shared<ALIPDDP<double>>(*problem_);
        solver_->init(solver_params_);
        solver_->solve();

        auto X_result = solver_->getResX();
        auto U_result = solver_->getResU();

        result.solve_time_ms   = chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
        result.solve_timestamp = chrono::steady_clock::now();
        last_solve_ms_         = result.solve_time_ms;

        if (X_result.size() > 1) {
            result.next_state         = X_result[1];
            result.state_trajectory   = X_result;
            result.control_trajectory = U_result;
            result.success            = true;
            prev_X_                   = X_result;
            prev_U_                   = U_result;
            std::cout << "[MPC] solve took " << result.solve_time_ms
                      << "ms  n_shift=" << config_.n_shift << "\n";
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
        last_solve_ms_ = 0.0;
    }
}