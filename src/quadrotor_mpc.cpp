#include "quadrotor_mpc.hpp"
#include "ocp_registry.hpp"
#include <iostream>

using namespace std;

QuadrotorMPC::QuadrotorMPC(const Config& config) : config_(config) {
    // Solver parameters are defined per-OCP in ocp_registry.hpp.
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
        return config_.dt;  // fallback
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
// Set up (or refresh) the OCP problem and inject the warm-start.
//
// KEY DESIGN: when should we call create() vs reuse the existing problem_?
//
//   create() builds the entire OCP: dynamics, cost structure, constraints,
//   and possibly a reference trajectory interpolated from current_state to
//   terminal_state (depending on the OCP implementation).
//
//   If create() re-interpolates a reference trajectory from current_state,
//   calling it on every solve changes the cost landscape every tick. The
//   warm-start U from the previous solve is optimal for the OLD landscape —
//   so the solver ignores it and cold-starts, producing jagged paths.
//
//   The fix: call create() ONLY when the problem genuinely changes:
//     - First solve (cold start)
//     - Terminal state changed (setTerminalState() called)
//     - OCP type changed
//
//   On subsequent solves we reuse problem_ and only update x[0] and U.
//   The cost landscape is identical to the previous solve, so the warm-start
//   U is a valid and good initial guess.
//
// NOTE: After reusing problem_, we still recreate the solver object because
//   ALIPDDP stores internal workspace tied to one problem instance. Recreating
//   the solver but reusing the problem_ pointer is safe as long as the problem
//   object outlives the solver.
// ─────────────────────────────────────────────────────────────────────────────
void QuadrotorMPC::setupProblem(const Eigen::VectorXd& current_state)
{
    // ── Rebuild the problem only when necessary ───────────────────────────────
    if (!problem_ || need_problem_rebuild_) {
        try {
            // Pass prev_U_ so stage costs get proper u_prev[k] for slew-rate
            // penalty. On cold start prev_U_ is empty → create() defaults to
            // u_ref (hover) for all k.
            problem_ = OCPRegistry::create(
                config_.ocp_type, current_state, config_.terminal_state, prev_U_);
        } catch (const std::runtime_error& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            throw;
        }
        need_problem_rebuild_ = false;
        solver_.reset();  // force solver re-init on next solve() call since problem changed
        // create() already seeded U with a gravity-compensating hover from the
        // current quaternion and called setInitialState(0, current_state).
        // On a cold start we leave that seed as-is.
        return;
    }

    // ── Reuse existing problem — only update x[0] and U ──────────────────────
    //
    // DDP reads only x[0] from outside. x[1..N] are always recomputed by the
    // solver's own forward rollout before the first backward pass.
    problem_->setInitialState(0, current_state);

    // Shifting and warm-starting are handled in solve() after this returns.
}

// ─────────────────────────────────────────────────────────────────────────────
QuadrotorMPC::Result QuadrotorMPC::solve(const Eigen::VectorXd& current_state)
{
    Result result;
    result.success = false;
    auto t0 = chrono::high_resolution_clock::now();

    try {
        // ── Project state onto OCP model manifold ─────────────────────────────
        Eigen::VectorXd x0_ocp = current_state;
        x0_ocp.segment(10, 3).setZero();   // zero wx, wy, wz

        setupProblem(x0_ocp);   // <-- pass x0_ocp, not current_state

        if (!solver_) {
            solver_ = make_shared<ALIPDDP<double>>(*problem_);
            solver_->init(solver_params_);
        } else {
            // Log the PHYSICAL state for diagnosis, but warm-start with projected
            std::cout << "warmstart x0 (raw):  " << current_state.transpose() << "\n";
            std::cout << "warmstart x0 (ocp):  " << x0_ocp.transpose() << "\n";
            if (prev_X_.size() > 1) {
                const Eigen::VectorXd dx_pos = current_state.head(6) - prev_X_[1].head(6);
                std::cout << "||pos+vel mismatch||: " << dx_pos.norm() << "\n";
            }

            // Shift U forward by n_shift steps.
            shiftWarmStart();

            // Roll dynamics forward from x0_ocp using shifted U so that
            // the warm-start X is consistent with the measured state.
            {
                auto dyn = std::make_shared<Quad6DOF<double>>();
                dyn->setMass(LandingOCP::MASS);
                dyn->setGravity(Eigen::Vector3d(0, 0, -9.81));
                dyn->setJb(LandingOCP::INERTIA);
                dyn->setDt(LandingOCP::DT);

                Eigen::VectorXd xr = x0_ocp;
                std::vector<Eigen::VectorXd> X_ws;
                X_ws.reserve(LandingOCP::HORIZON + 1);
                X_ws.push_back(xr);
                for (int k = 0; k < LandingOCP::HORIZON; ++k) {
                    xr = dyn->f(xr, prev_U_[k]);
                    X_ws.push_back(xr);
                }
                prev_X_ = X_ws;  // update cached X to match
            }

            // warmStart takes (x0, U): injects x0 and U into the solver's
            // internal arrays without resetting AL multipliers.
            solver_->init(solver_params_);          // resets λ, ρ to initial values
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
        // Force create() to be called on the next solve so the new terminal
        // is reflected in the cost function.
        need_problem_rebuild_ = true;
    }
}