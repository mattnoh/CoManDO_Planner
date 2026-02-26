#include "quadrotor_mpc.hpp"
#include "ocp_hover.hpp"
#include "ocp_landing.hpp"
#include <iostream>

using namespace std;

QuadrotorMPC::QuadrotorMPC(const Config& config) : config_(config) {
    // Solver parameters are defined per-OCP in the header files.
    // Each OCP tunes these independently: hover needs fewer iterations and
    // lower rho (simple quadratic, one inequality), landing needs higher rho
    // and more iterations (SOC constraints + soft terminal cost).
    if (config_.ocp_type == "hover") {
        solver_params_.reg1_min  = HoverOCP::SOLVER_REG1_MIN;
        solver_params_.reg2_min  = HoverOCP::SOLVER_REG2_MIN;
        solver_params_.mu_mul    = HoverOCP::SOLVER_MU_MUL;
        solver_params_.rho       = HoverOCP::SOLVER_RHO;
        solver_params_.rho_mul   = HoverOCP::SOLVER_RHO_MUL;
        solver_params_.tolerance = HoverOCP::SOLVER_TOLERANCE;
        solver_params_.max_iter  = HoverOCP::SOLVER_MAX_ITER;
    } else if (config_.ocp_type == "landing") {
        solver_params_.reg1_min  = LandingOCP::SOLVER_REG1_MIN;
        solver_params_.reg2_min  = LandingOCP::SOLVER_REG2_MIN;
        solver_params_.mu_mul    = LandingOCP::SOLVER_MU_MUL;
        solver_params_.rho       = LandingOCP::SOLVER_RHO;
        solver_params_.rho_mul   = LandingOCP::SOLVER_RHO_MUL;
        solver_params_.tolerance = LandingOCP::SOLVER_TOLERANCE;
        solver_params_.max_iter  = LandingOCP::SOLVER_MAX_ITER;
    } else {
        // Fallback defaults
        solver_params_.reg1_min  = 1e-6;
        solver_params_.reg2_min  = 1.0;
        solver_params_.mu_mul    = 0.1;
        solver_params_.rho       = 20.0;
        solver_params_.rho_mul   = 9.0;
        solver_params_.tolerance = 1e-3;
        solver_params_.max_iter  = 200;
    }
}

double QuadrotorMPC::getOcpDt() const {
    if (config_.ocp_type == "hover")   return HoverOCP::DT;
    if (config_.ocp_type == "landing") return LandingOCP::DT;
    return config_.dt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shift the warm-start solution forward by n_shift steps.
//
// WHY n_shift AND NOT ALWAYS 1:
//   The solver fires every (1/solver_rate) seconds. The OCP time step is ocp_dt.
//   Between two solver ticks the drone has consumed:
//
//     n_shift = round( solver_period / ocp_dt )
//
//   steps of the trajectory. If solver_rate=10Hz and ocp_dt=0.05s, n_shift=2.
//   Shifting by only 1 leaves the warm-start one step behind where the drone
//   actually is — after a few solves the reference drifts, the solver has to
//   work harder each time to reconcile, and the published path looks jagged
//   as if solving from cold each time.
//
// WHAT WE DO WITH prev_X_:
//   Shifted but NOT passed to the solver. DDP always recomputes x[1..N] from
//   x[0] and U in its own forward rollout — x[1..N] set from outside are
//   immediately overwritten. Kept for future use if we add an X-seed API.
//
// WHAT WE DO WITH prev_U_:
//   This IS the warm-start seed. After shifting by n_shift:
//     u_warm[k] = u_prev[k + n_shift]   for k = 0 .. N-1-n_shift
//     u_warm[k] = u_prev[N-1]           for k = N-n_shift .. N-1  (hold last)
// ─────────────────────────────────────────────────────────────────────────────
void QuadrotorMPC::shiftWarmStart() {
    if (prev_X_.size() < 2 || prev_U_.empty()) return;
    const int Nx = (int)prev_X_.size();
    const int Nu = (int)prev_U_.size();
    const int n_shift = std::max(1, std::min(config_.n_shift, Nu - 1));

    // Shift X — NOT sent to the solver, kept for future use only.
    {
        vector<Eigen::VectorXd> sx(Nx);
        for (int i = 0; i < Nx; ++i)
            sx[i] = prev_X_[std::min(i + n_shift, Nx - 1)];
        prev_X_ = move(sx);
    }

    // Shift U — this IS the DDP warm-start seed.
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

    if (has_prev_solution_) {
        // Shift U forward by n_shift steps (one step per ocp_dt elapsed).
        // This overrides whatever the problem currently holds, which is what
        // we want — the previous optimal solution is a much better seed than
        // the generic hover create() provided on cold start.
        //
        // REMOVED (old code): manual forward rollout setting x[1..N] via
        //   getDynamics(i)->f(x, u) → setInitialState(i+1, ...).
        // DDP overwrites all of those in its first forward sweep anyway.
        // shiftWarmStart();

        // for (int i = 0; i < (int)prev_U_.size(); ++i)
        //     problem_->setInitialControl(i, prev_U_[i]);
    }
    // If !has_prev_solution_ here, the problem still holds the create() hover
    // seed from the first call — which is a valid cold-start U.
}

// ─────────────────────────────────────────────────────────────────────────────
QuadrotorMPC::Result QuadrotorMPC::solve(const Eigen::VectorXd& current_state)
{
    Result result;
    result.success = false;
    auto t0 = chrono::high_resolution_clock::now();

    try {
        setupProblem(current_state);

        if (!problem_) {
            cerr << "ERROR: Problem not initialized\n";
            return result;
        }

        // ── Create and init solver ONCE per problem lifetime ──────────────────
        // Recreating the solver every tick resets AL multipliers (lambda, Y, S,
        // Z, R) to zero, forcing a cold start every iteration even with a good
        // U warm-start. Persisting the solver lets the multipliers carry over
        // so subsequent solves start near the previous solution.
        //
        // NOTE: the solver's internal ocp is a copy made at construction time —
        // calling problem_->setInitialState/Control() after construction has no
        // effect on the solver. We must use warmStart() to inject x[0] and U
        // directly into the solver's internal X[0] and U arrays.
        if (!solver_) {
            solver_ = make_shared<ALIPDDP<double>>(*problem_);
            solver_->init(solver_params_);
        } else {
            // Shift warm-start U forward by n_shift steps, then inject
            // x[0] and U directly into the solver without touching multipliers.
            shiftWarmStart();
            solver_->warmStart(current_state, prev_U_);
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

            // Cache full solution for the next warm start.
            // prev_X_ is not read by the solver — stored for future use only.
            // prev_U_ is the actual warm-start seed shifted and passed to the
            // next setupProblem() call.
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
        has_prev_solution_    = false;
        // Force create() to be called on the next solve so the new terminal
        // is reflected in the cost function.
        need_problem_rebuild_ = true;
    }
}