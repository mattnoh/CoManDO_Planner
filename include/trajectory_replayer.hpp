/// @file trajectory_replayer.hpp
/// @brief Thread-safe, time-continuous trajectory sampler for async replanning.
///
/// Key design: the replay timer and solver run on independent cadences.
/// The replayer stores the latest trajectory and samples it by wall-clock
/// time with linear interpolation. When a new solve arrives, the buffer
/// is swapped atomically. The replay timer never "waits" for a solve.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <mutex>
#include <optional>
#include <vector>

class TrajectoryReplayer {
public:
    TrajectoryReplayer() = default;

    void clear() {
        std::lock_guard<std::mutex> lk(mutex_);
        state_trajectory_.clear();
        control_trajectory_.clear();
        active_solve_num_ = -1;
        has_variable_dt_ = false;
        state_dim_ = 13;
        last_solve_wall_time_ = std::chrono::steady_clock::time_point{};
        pending_state_trajectory_.clear();
        pending_control_trajectory_.clear();
        pending_solve_num_ = -1;
        pending_has_variable_dt_ = false;
        pending_state_dim_ = 13;
        pending_solve_wall_time_ = std::chrono::steady_clock::time_point{};
        pending_earliest_activation_time_ = std::chrono::steady_clock::time_point{};
        pending_solve_latency_sec_ = 0.0;
        has_pending_plan_ = false;
        min_replay_before_swap_ = 1;
        last_handoff_diagnostic_.reset();
    }

    struct ReplaySample {
        bool has_plan = false;
        int active_solve_num = -1;
        double elapsed = 0.0;
        double horizon_end = 0.0;
        Eigen::VectorXd x_cmd;
        Eigen::VectorXd x_cmd_lookahead;
        Eigen::VectorXd u_cmd;
    };

    struct HandoffDiagnostic {
        bool swapped = false;
        int previous_solve_num = -1;
        int new_solve_num = -1;
        double active_plan_age_sec = 0.0;
        double solve_latency_sec = 0.0;
        double state_jump_norm = 0.0;
        double control_jump_norm = 0.0;
        Eigen::VectorXd old_state_cmd;
        Eigen::VectorXd new_state_cmd;
        Eigen::VectorXd old_control_cmd;
        Eigen::VectorXd new_control_cmd;
    };

    /// Push a new trajectory into the buffer.
    /// plan_origin_time is the wall-clock time represented by trajectory node 0.
    /// earliest_activation_time is the first wall-clock time at which this plan may replace
    /// the current active plan. For the first plan both should normally be solve_finish_time.
    /// state_dim: physical state dimension (e.g. 13 for standard, 22 for body-frame).
    /// For variable-dt OCPs the DT element lives at index state_dim in each state vector.
    void updatePlan(const std::vector<Eigen::VectorXd>& state_trajectory,
                    const std::vector<Eigen::VectorXd>& control_trajectory,
                    double solve_time_ms,
                    int solve_num,
                    const std::chrono::steady_clock::time_point& plan_origin_time,
                    const std::chrono::steady_clock::time_point& earliest_activation_time,
                    double /*ocp_dt*/,
                    int min_replay_before_swap,
                    bool is_variable_dt = false,
                    int state_dim = 13) {
        std::lock_guard<std::mutex> lk(mutex_);
        min_replay_before_swap_ = std::max(1, min_replay_before_swap);

        // First accepted solve must become active immediately to avoid command gaps.
        if (state_trajectory_.empty()) {
            state_trajectory_ = state_trajectory;
            control_trajectory_ = control_trajectory;
            active_solve_num_ = solve_num;
            last_solve_wall_time_ = plan_origin_time;
            has_variable_dt_ = is_variable_dt;
            state_dim_ = state_dim;
            return;
        }

        // Keep only the freshest pending plan; swap-in decision happens in sample().
        pending_state_trajectory_ = state_trajectory;
        pending_control_trajectory_ = control_trajectory;
        pending_solve_num_ = solve_num;
        pending_solve_wall_time_ = plan_origin_time;
        pending_earliest_activation_time_ = earliest_activation_time;
        pending_solve_latency_sec_ = 0.001 * solve_time_ms;
        pending_has_variable_dt_ = is_variable_dt;
        pending_state_dim_ = state_dim;
        has_pending_plan_ = true;
    }

    std::optional<HandoffDiagnostic> consumeLastHandoffDiagnostic() {
        std::lock_guard<std::mutex> lk(mutex_);
        auto diag = last_handoff_diagnostic_;
        last_handoff_diagnostic_.reset();
        return diag;
    }

    bool hasPendingPlan() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return has_pending_plan_;
    }

    std::chrono::steady_clock::time_point activePlanOriginTime() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return last_solve_wall_time_;
    }

    double activeElapsedAt(const std::chrono::steady_clock::time_point& now) const {
        std::lock_guard<std::mutex> lk(mutex_);
        if (state_trajectory_.empty()) {
            return 0.0;
        }
        return std::chrono::duration<double>(now - last_solve_wall_time_).count();
    }

    double activeHorizonEnd(double ocp_dt) const {
        std::lock_guard<std::mutex> lk(mutex_);
        const int active_N = static_cast<int>(state_trajectory_.size()) - 1;
        return (active_N > 0) ? nodeTime(active_N, ocp_dt) : 0.0;
    }

    bool sampleActiveAtElapsed(double elapsed, double ocp_dt, ReplaySample* out) const {
        if (out == nullptr) {
            return false;
        }
        std::lock_guard<std::mutex> lk(mutex_);
        if (state_trajectory_.empty()) {
            return false;
        }
        *out = makeSampleLocked(state_trajectory_, control_trajectory_, active_solve_num_,
                                has_variable_dt_, state_dim_, elapsed, ocp_dt);
        return out->has_plan;
    }

    /// Return the first active-trajectory node at or after `elapsed`.
    /// Handoffs scheduled on this boundary can use the same integer shift for
    /// both the predicted x0 and the solver warm start.
    bool activeNodeAtOrAfter(double elapsed, double ocp_dt,
                             int* node_index, double* node_time) const {
        std::lock_guard<std::mutex> lk(mutex_);
        if (state_trajectory_.size() < 2) return false;
        const int N = static_cast<int>(state_trajectory_.size()) - 1;
        for (int k = 1; k <= N; ++k) {
            const double tk =
                nodeTimeFrom(state_trajectory_, has_variable_dt_, state_dim_, k, ocp_dt);
            if (tk + 1e-9 >= elapsed) {
                if (node_index) *node_index = k;
                if (node_time) *node_time = tk;
                return true;
            }
        }
        return false;
    }

    /// Sample the active trajectory at the current wall-clock time.
    /// Returns an interpolated setpoint between the bracketing nodes.
    ReplaySample sample(const std::chrono::steady_clock::time_point& now,
                        double ocp_dt) {
        ReplaySample s;
        s.x_cmd = Eigen::VectorXd::Zero(state_dim_);
        s.u_cmd = Eigen::VectorXd::Zero(4);

        std::lock_guard<std::mutex> lk(mutex_);

        maybeSwapPendingPlan(now, ocp_dt);

        if (state_trajectory_.empty()) {
            return s;
        }

        s.has_plan = true;
        s.active_solve_num = active_solve_num_;
        s.elapsed = std::chrono::duration<double>(now - last_solve_wall_time_).count();

        return makeSampleLocked(state_trajectory_, control_trajectory_, active_solve_num_,
                                has_variable_dt_, state_dim_, s.elapsed, ocp_dt);
    }

private:
    int findLowerNodeFrom(const std::vector<Eigen::VectorXd>& states,
                          bool variable_dt,
                          int state_dim,
                          double t_sample,
                          int N,
                          double ocp_dt) const {
        int k_lo = 0;
        for (int k = 0; k < N; ++k) {
            if (nodeTimeFrom(states, variable_dt, state_dim, k + 1, ocp_dt) > t_sample) {
                k_lo = k;
                break;
            }
            k_lo = k;
        }
        return k_lo;
    }

    Eigen::VectorXd sampleStateFrom(const std::vector<Eigen::VectorXd>& states,
                                    bool variable_dt,
                                    int state_dim,
                                    double t_sample,
                                    double ocp_dt) const {
        const int N = static_cast<int>(states.size()) - 1;
        if (N < 1) {
            return Eigen::VectorXd::Zero(state_dim);
        }

        const int k_lo = findLowerNodeFrom(states, variable_dt, state_dim, t_sample, N, ocp_dt);
        const int k_hi = std::min(k_lo + 1, N);

        double alpha = 0.0;
        if (k_lo < k_hi) {
            const double dt_seg =
                nodeTimeFrom(states, variable_dt, state_dim, k_hi, ocp_dt) -
                nodeTimeFrom(states, variable_dt, state_dim, k_lo, ocp_dt);
            if (dt_seg > 1e-9) {
                alpha = std::clamp(
                    (t_sample - nodeTimeFrom(states, variable_dt, state_dim, k_lo, ocp_dt)) /
                    dt_seg,
                    0.0, 1.0);
            }
        }

        const auto& x_lo = states[k_lo];
        const auto& x_hi = states[k_hi];

        // Allocate full-sized output (physical states + optional DT slot)
        const int full_len = static_cast<int>(x_lo.size());
        const int interp_len = std::min(state_dim, std::min((int)x_lo.size(), (int)x_hi.size()));
        Eigen::VectorXd x_interp = x_lo;  // copy whole vector first (preserves DT and extras)

        // Linear interpolation for all physical states
        if (interp_len > 0) {
            x_interp.head(interp_len) =
                (1.0 - alpha) * x_lo.head(interp_len) + alpha * x_hi.head(interp_len);
        }

        // Quaternion re-normalisation [6:10] — undo linear damage
        if (interp_len >= 10) {
            Eigen::Vector4d q_lo = x_lo.segment(6, 4);
            Eigen::Vector4d q_hi = x_hi.segment(6, 4);
            if (q_lo.dot(q_hi) < 0.0) q_hi = -q_hi;
            Eigen::Vector4d q_interp = (1.0 - alpha) * q_lo + alpha * q_hi;
            const double qn = q_interp.norm();
            if (qn > 1e-9) q_interp /= qn;
            x_interp.segment(6, 4) = q_interp;
        }

        // DT slot [state_dim_]: NOT interpolated — keep the k_lo value (already copied above).
        // Any trailing elements beyond DT are also kept from x_lo.
        (void)full_len;

        return x_interp;
    }

    Eigen::VectorXd sampleControlFrom(const std::vector<Eigen::VectorXd>& controls,
                                      const std::vector<Eigen::VectorXd>& states,
                                      bool variable_dt,
                                      int state_dim,
                                      double t_sample,
                                      double ocp_dt) const {
        if (controls.empty() || states.empty()) {
            return Eigen::VectorXd::Zero(4);
        }
        const int N = static_cast<int>(states.size()) - 1;
        const int k_lo = findLowerNodeFrom(states, variable_dt, state_dim, t_sample,
                                           std::max(0, N), ocp_dt);
        const int u_idx = std::min(k_lo, static_cast<int>(controls.size()) - 1);
        const auto& u_full = controls[u_idx];
        return u_full.head(std::min(static_cast<int>(u_full.size()), 4));
    }

    ReplaySample makeSampleLocked(const std::vector<Eigen::VectorXd>& states,
                                  const std::vector<Eigen::VectorXd>& controls,
                                  int solve_num,
                                  bool variable_dt,
                                  int state_dim,
                                  double elapsed,
                                  double ocp_dt) const {
        ReplaySample s;
        s.x_cmd = Eigen::VectorXd::Zero(state_dim);
        s.u_cmd = Eigen::VectorXd::Zero(4);
        if (states.empty()) {
            return s;
        }

        s.has_plan = true;
        s.active_solve_num = solve_num;
        s.elapsed = elapsed;

        const int N = static_cast<int>(states.size()) - 1;
        if (N < 1) {
            return s;
        }

        s.horizon_end = nodeTimeFrom(states, variable_dt, state_dim, N, ocp_dt);
        const double t_sample = std::clamp(s.elapsed, 0.0, s.horizon_end);
        const int k_lo = findLowerNodeFrom(states, variable_dt, state_dim, t_sample, N, ocp_dt);
        double t_lookahead = std::clamp(t_sample + ocp_dt, 0.0, s.horizon_end);
        if (variable_dt) {
            t_lookahead = nodeTimeFrom(states, variable_dt, state_dim, std::min(k_lo + 1, N),
                                       ocp_dt);
        }

        s.x_cmd = sampleStateFrom(states, variable_dt, state_dim, t_sample, ocp_dt);
        s.x_cmd_lookahead = sampleStateFrom(states, variable_dt, state_dim, t_lookahead, ocp_dt);
        s.u_cmd = sampleControlFrom(controls, states, variable_dt, state_dim, t_sample, ocp_dt);
        return s;
    }

    void maybeSwapPendingPlan(const std::chrono::steady_clock::time_point& now,
                              double ocp_dt) {
        if (!has_pending_plan_) {
            return;
        }

        const double elapsed = std::chrono::duration<double>(now - last_solve_wall_time_).count();
        const int active_N = static_cast<int>(state_trajectory_.size()) - 1;
        const double horizon_end = (active_N > 0) ? nodeTime(active_N, ocp_dt) : 0.0;
        const bool active_stale = (active_N <= 0) || (elapsed > horizon_end + 0.2);

        const int advance_idx = std::clamp(min_replay_before_swap_, 0, std::max(0, active_N));
        const double min_hold_time = (advance_idx > 0) ? nodeTime(advance_idx, ocp_dt) : 0.0;
        const bool min_replay_elapsed = (elapsed >= min_hold_time);
        const bool activation_time_reached = (now >= pending_earliest_activation_time_);

        if (!activation_time_reached || (!active_stale && !min_replay_elapsed)) {
            return;
        }

        const double old_elapsed = std::chrono::duration<double>(now - last_solve_wall_time_).count();
        const double new_elapsed = std::chrono::duration<double>(now - pending_solve_wall_time_).count();
        const auto old_sample = makeSampleLocked(state_trajectory_, control_trajectory_,
                                                 active_solve_num_, has_variable_dt_, state_dim_,
                                                 old_elapsed, ocp_dt);
        const auto new_sample = makeSampleLocked(pending_state_trajectory_,
                                                 pending_control_trajectory_, pending_solve_num_,
                                                 pending_has_variable_dt_, pending_state_dim_,
                                                 new_elapsed, ocp_dt);

        HandoffDiagnostic diag;
        diag.swapped = true;
        diag.previous_solve_num = active_solve_num_;
        diag.new_solve_num = pending_solve_num_;
        diag.active_plan_age_sec = old_elapsed;
        diag.solve_latency_sec = pending_solve_latency_sec_;
        diag.old_state_cmd = old_sample.x_cmd;
        diag.new_state_cmd = new_sample.x_cmd;
        diag.old_control_cmd = old_sample.u_cmd;
        diag.new_control_cmd = new_sample.u_cmd;
        if (diag.old_state_cmd.size() == diag.new_state_cmd.size() && diag.old_state_cmd.size() > 0) {
            diag.state_jump_norm = (diag.new_state_cmd - diag.old_state_cmd).norm();
        }
        if (diag.old_control_cmd.size() == diag.new_control_cmd.size() &&
            diag.old_control_cmd.size() > 0) {
            diag.control_jump_norm = (diag.new_control_cmd - diag.old_control_cmd).norm();
        }

        state_trajectory_ = pending_state_trajectory_;
        control_trajectory_ = pending_control_trajectory_;
        active_solve_num_ = pending_solve_num_;
        last_solve_wall_time_ = pending_solve_wall_time_;
        has_variable_dt_ = pending_has_variable_dt_;
        state_dim_ = pending_state_dim_;

        pending_state_trajectory_.clear();
        pending_control_trajectory_.clear();
        pending_solve_num_ = -1;
        pending_has_variable_dt_ = false;
        pending_state_dim_ = 13;
        pending_solve_wall_time_ = std::chrono::steady_clock::time_point{};
        pending_earliest_activation_time_ = std::chrono::steady_clock::time_point{};
        pending_solve_latency_sec_ = 0.0;
        has_pending_plan_ = false;
        last_handoff_diagnostic_ = diag;
    }

    double nodeTime(int k, double ocp_dt) const {
        return nodeTimeFrom(state_trajectory_, has_variable_dt_, state_dim_, k, ocp_dt);
    }

    double nodeTimeFrom(const std::vector<Eigen::VectorXd>& states,
                        bool variable_dt,
                        int state_dim,
                        int k,
                        double ocp_dt) const {
        if (variable_dt &&
            k < static_cast<int>(states.size()) &&
            states[k].size() > state_dim) {
            return states[k](state_dim);
        }
        return k * ocp_dt;
    }

    mutable std::mutex mutex_;
    std::vector<Eigen::VectorXd> state_trajectory_;
    std::vector<Eigen::VectorXd> control_trajectory_;
    int active_solve_num_ = -1;
    bool has_variable_dt_ = false;
    int state_dim_ = 13;  ///< Physical state dimension; DT slot lives at index state_dim_
    std::chrono::steady_clock::time_point last_solve_wall_time_{};

    std::vector<Eigen::VectorXd> pending_state_trajectory_;
    std::vector<Eigen::VectorXd> pending_control_trajectory_;
    int pending_solve_num_ = -1;
    bool pending_has_variable_dt_ = false;
    int pending_state_dim_ = 13;
    std::chrono::steady_clock::time_point pending_solve_wall_time_{};
    std::chrono::steady_clock::time_point pending_earliest_activation_time_{};
    double pending_solve_latency_sec_ = 0.0;
    bool has_pending_plan_ = false;
    int min_replay_before_swap_ = 1;
    std::optional<HandoffDiagnostic> last_handoff_diagnostic_;
};
