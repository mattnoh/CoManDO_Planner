#include "planner_core/quadrotor_mpc.hpp"
#include "planner_core/ocp_registry.hpp"

#include <algorithm>
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
    return OCPRegistry::getDT(config_.ocp_type);
}

QuadrotorMPC::WarmStartSnapshot QuadrotorMPC::snapshotWarmStart() const {
    WarmStartSnapshot snapshot;
    snapshot.prev_X = prev_X_;
    snapshot.prev_U = prev_U_;
    snapshot.prev_K = prev_K_;
    snapshot.last_solve_ms = last_solve_ms_;
    return snapshot;
}

void QuadrotorMPC::restoreWarmStart(const WarmStartSnapshot& snapshot) {
    prev_X_ = snapshot.prev_X;
    prev_U_ = snapshot.prev_U;
    prev_K_ = snapshot.prev_K;
    last_solve_ms_ = snapshot.last_solve_ms;
    next_warm_start_unshifted_ = false;
    next_warm_start_shift_ = -1;
}

void QuadrotorMPC::keepLatestAsUnexecutedWarmStart() {
    next_warm_start_unshifted_ = true;
}

std::vector<Eigen::VectorXd> QuadrotorMPC::makeUwarm(int n_shift) const {
    if (prev_U_.empty()) return {};

    const int N = static_cast<int>(prev_U_.size());
    if (N == 1) return prev_U_;

    const int NEX = std::max(0, std::min(n_shift, N - 1));
    const auto desc = OCPRegistry::getDescriptor(config_.ocp_type);
    if (NEX == 0) {
        std::vector<Eigen::VectorXd> uw = prev_U_;
        for (auto& u : uw) {
            if (desc.sanitize_warm_control) {
                desc.sanitize_warm_control(u);
            }
        }
        return uw;
    }

    Eigen::VectorXd u_tail = prev_U_[N - 1 - NEX];
    if (u_tail.size() >= 4) {
        u_tail(1) = 0.0;
        u_tail(2) = 0.0;
        u_tail(3) = 0.0;
    }
    if (desc.sanitize_warm_control) {
        desc.sanitize_warm_control(u_tail);
    }

    std::vector<Eigen::VectorXd> uw(N);
    for (int i = 0; i < N; ++i) {
        if (i + NEX < N - NEX) {
            uw[i] = prev_U_[i + NEX];
        } else {
            uw[i] = u_tail;
        }
        if (desc.sanitize_warm_control) {
            desc.sanitize_warm_control(uw[i]);
        }
    }
    return uw;
}

std::vector<Eigen::VectorXd> QuadrotorMPC::makeXshifted(int n_shift) const {
    if (prev_X_.empty()) return {};

    const int Nx = static_cast<int>(prev_X_.size());
    const int NEX = std::max(0, n_shift);

    std::vector<Eigen::VectorXd> xs(Nx);
    for (int i = 0; i < Nx; ++i) {
        xs[i] = prev_X_[std::min(i + NEX, Nx - 1)];
    }
    return xs;
}

std::vector<Eigen::MatrixXd> QuadrotorMPC::makeKshifted(int n_shift) const {
    if (prev_K_.empty()) return {};

    const int Nk = static_cast<int>(prev_K_.size());
    const int NEX = std::max(0, n_shift);

    std::vector<Eigen::MatrixXd> ks(Nk);
    for (int i = 0; i < Nk; ++i) {
        ks[i] = prev_K_[std::min(i + NEX, Nk - 1)];
    }
    return ks;
}

void QuadrotorMPC::setupProblem(const OCPCreateArgs& args) {
    try {
        solver_.reset();
        problem_ = OCPRegistry::create(config_.ocp_type, args);
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        throw;
    }
}

QuadrotorMPC::Result QuadrotorMPC::solve(const Eigen::VectorXd& current_state,
                                           const Eigen::Vector3d& target_accel,
                                           const std::any& extra_params,
                                           double t_abs)
{
    Result result;
    result.success = false;
    auto t0 = chrono::steady_clock::now();
    result.solve_start_time = t0;
    result.solve_timestamp = t0;

    try {
        target_accel_ = target_accel;

        std::vector<Eigen::VectorXd> warm_u = prev_U_;
        std::vector<Eigen::VectorXd> warm_x = prev_X_;
        std::vector<Eigen::MatrixXd> warm_k = prev_K_;

        if (!prev_U_.empty()) {
            auto desc = OCPRegistry::getDescriptor(config_.ocp_type);
            const int warm_shift = next_warm_start_unshifted_ ? 0 :
                (next_warm_start_shift_ >= 0 ? next_warm_start_shift_ : config_.n_shift);
            if (desc.warm_start == OCPDescriptor::WarmStart::Feedback) {
                warm_u = makeUwarm(warm_shift);
                warm_x = makeXshifted(warm_shift);
                warm_k = makeKshifted(warm_shift);
            } else {
                const int Nu = static_cast<int>(prev_U_.size());
                const int Nx = static_cast<int>(prev_X_.size());
                const int shift = std::max(0, std::min(warm_shift, std::max(1, Nu - 1)));

                std::vector<Eigen::VectorXd> su(Nu);
                for (int i = 0; i < Nu; ++i) {
                    su[i] = prev_U_[std::min(i + shift, Nu - 1)];
                }
                warm_u = std::move(su);

                if (Nx > 0) {
                    std::vector<Eigen::VectorXd> sx(Nx);
                    for (int i = 0; i < Nx; ++i) {
                        sx[i] = prev_X_[std::min(i + shift, Nx - 1)];
                    }
                    warm_x = std::move(sx);
                }
            }
        }

        OCPCreateArgs args;
        args.current_state = current_state;
        args.terminal_state = config_.terminal_state;
        args.prev_U = warm_u;
        args.prev_X = warm_x;
        args.prev_K = warm_k;
        args.target_accel = target_accel;
        args.extra = extra_params;
        args.t_abs = t_abs;

        setupProblem(args);

        solver_ = make_shared<ALIPDDP<double>>(*problem_);
        solver_->init(solver_params_);
        solver_->solve();

        auto X_result = solver_->getResX();
        auto U_result = solver_->getResU();
        auto K_result = solver_->getResK();

        const auto t_finish = chrono::steady_clock::now();
        result.solve_finish_time = t_finish;
        result.solve_time_ms = chrono::duration<double, milli>(
            t_finish - t0).count();
        result.solve_timestamp = result.solve_start_time;
        result.constraint_error = solver_->getError();
        result.solve_iters = static_cast<int>(solver_->getAllCost().size());
        result.extra_params = extra_params;
        last_solve_ms_ = result.solve_time_ms;

        if (X_result.size() > 1) {
            auto desc = OCPRegistry::getDescriptor(config_.ocp_type);
            int state_dim = desc.state_dim;
            if (X_result[1].size() >= state_dim) {
                result.next_state = X_result[1].head(state_dim);
            } else {
                result.next_state = X_result[1];
            }
            result.state_trajectory = X_result;
            result.control_trajectory = U_result;
            result.feedback_gains = K_result;
            result.success = true;

            prev_X_ = X_result;
            prev_U_ = U_result;
            prev_K_ = K_result;
            next_warm_start_unshifted_ = false;
            next_warm_start_shift_ = -1;

            std::cout << "[MPC] solve " << result.solve_time_ms
                      << "ms iters=" << result.solve_iters
                      << " n_shift=" << config_.n_shift << "\n";
        } else {
            std::cerr << "ERROR: Empty trajectory\n";
            last_solve_ms_ = 0.0;
        }

    } catch (const exception& e) {
        std::cerr << "ERROR in solve: " << e.what() << "\n";
        last_solve_ms_ = 0.0;
        result.solve_finish_time = chrono::steady_clock::now();
    }

    return result;
}

void QuadrotorMPC::setTerminalState(const Eigen::VectorXd& terminal) {
    if (terminal.size() > 0) {
        config_.terminal_state = terminal;
        solver_.reset();
        prev_U_.clear();
        prev_X_.clear();
        prev_K_.clear();
        last_solve_ms_ = 0.0;
    }
}
