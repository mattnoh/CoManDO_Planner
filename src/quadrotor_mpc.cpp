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
// setupProblem now receives the shifted prev_X_ so the OCP factory can use
// it as the trajectory-consistency reference.
//
// On the first solve prev_X_ is empty — the factory detects this and disables
// the consistency penalty automatically (no-op, pure cold start).
//
// On subsequent solves prev_X_ has already been shifted by n_shift in
// shiftWarmStart(), so prev_X_[k] aligns temporally with node k of the
// new solve.
// ─────────────────────────────────────────────────────────────────────────────
void QuadrotorMPC::setupProblem(const Eigen::VectorXd& current_state)
{
    if (!problem_ || need_problem_rebuild_) {
        try {
            // Pass prev_X_ as the trajectory reference.
            // Empty on first solve → no consistency penalty.
            problem_ = OCPRegistry::create(
                config_.ocp_type, current_state, config_.terminal_state,
                prev_U_, prev_X_);
        } catch (const std::runtime_error& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            throw;
        }
        need_problem_rebuild_ = false;
        solver_.reset();
        return;
    }

    // Rebuild the problem every solve so that stage costs get the fresh
    // prev_X_ trajectory reference.  We keep the solver object alive for
    // warm-starting but reconstruct the problem (cost functions only).
    //
    // Alternative: expose a setCosts() method on the problem.  For now,
    // full rebuild is simpler and takes <1ms on a 100-node horizon.
    try {
        problem_ = OCPRegistry::create(
            config_.ocp_type, current_state, config_.terminal_state,
            prev_U_, prev_X_);
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        throw;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// γ-blend warm-start (Zhang et al. 2023, §III-B)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double VEL_ERR_SCALE = 0.3;
static constexpr double CRAZYFLIE_MASS = 0.027;

static void gammaBlend(std::vector<Eigen::VectorXd>& U_warm,
                       double vel_err,
                       double gamma)
{
    Eigen::VectorXd u_cold(4);
    u_cold << CRAZYFLIE_MASS * 9.81, 0.0, 0.0, 0.0;

    std::cout << "[gamma_blend] vel_err=" << vel_err
              << " gamma=" << gamma << "\n";

    for (auto& u : U_warm)
        u = gamma * u + (1.0 - gamma) * u_cold;
}

// ─────────────────────────────────────────────────────────────────────────────
QuadrotorMPC::Result QuadrotorMPC::solve(const Eigen::VectorXd& current_state)
{
    Result result;
    result.success = false;
    auto t0 = chrono::high_resolution_clock::now();

    try {
        Eigen::VectorXd x0_ocp = current_state;
        x0_ocp.segment(10, 3).setZero();

        if (!solver_) {
            // ── First solve: cold start ───────────────────────────────────────
            // prev_X_ is empty → consistency penalty disabled in factory.
            setupProblem(x0_ocp);
            solver_ = make_shared<ALIPDDP<double>>(*problem_);
            solver_->init(solver_params_);

        } else {
            // ── Subsequent solves ─────────────────────────────────────────────

            std::cout << "x0_actual:  " << current_state.transpose() << "\n";
            std::cout << "prev_X_[1]: " << prev_X_[1].transpose() << "\n";
            const double pos_err  = (current_state.head(3)     - prev_X_[1].head(3)).norm();
            const double vel_err  = (current_state.segment(3,3) - prev_X_[1].segment(3,3)).norm();
            const double quat_dot = std::abs(current_state.segment(6,4).dot(prev_X_[1].segment(6,4)));
            const double att_err  = 2.0 * std::acos(std::min(1.0, quat_dot)) * 180.0/M_PI;
            std::cout << "pos_err=" << pos_err
                      << " vel_err=" << vel_err
                      << " att_err_deg=" << att_err << "\n";

            // ── Step 1: shift warm-start forward by n_shift steps ─────────────
            // After this, prev_X_[k] aligns with node k of the new solve.
            // The OCP factory (called in setupProblem below) uses this shifted
            // prev_X_ as the trajectory-consistency reference.
            shiftWarmStart();

            // ── Step 2: rebuild problem with fresh trajectory reference ────────
            // The stage costs now include ||x_k - prev_X_[k+1]||_W² terms.
            setupProblem(x0_ocp);

            // ── Step 3: γ-blend toward hover cold-start ───────────────────────
            {
                const double gamma = std::max(0.0, 1.0 - vel_err / VEL_ERR_SCALE);
                if (gamma < 0.999) {
                    gammaBlend(prev_U_, vel_err, gamma);
                }
            }

            // ── Step 4: warm-start the solver ─────────────────────────────────
            solver_->init(solver_params_);
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