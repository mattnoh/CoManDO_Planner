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
#include <vector>

class TrajectoryReplayer {
public:
    void clear() {
        std::lock_guard<std::mutex> lk(mutex_);
        state_trajectory_.clear();
        control_trajectory_.clear();
        active_solve_num_ = -1;
        plan_is_relative_ = false;
        has_variable_dt_ = false;
        last_solve_wall_time_ = std::chrono::steady_clock::time_point{};
        pending_state_trajectory_.clear();
        pending_control_trajectory_.clear();
        pending_solve_num_ = -1;
        pending_plan_is_relative_ = false;
        pending_has_variable_dt_ = false;
        pending_solve_wall_time_ = std::chrono::steady_clock::time_point{};
        has_pending_plan_ = false;
        min_replay_before_swap_ = 1;
    }

    struct ReplaySample {
        bool has_plan = false;
        int active_solve_num = -1;
        bool plan_is_relative = false;
        double elapsed = 0.0;
        double horizon_end = 0.0;
        Eigen::VectorXd x_cmd;
        Eigen::VectorXd x_cmd_lookahead;
        Eigen::VectorXd u_cmd;
        Eigen::VectorXd x_rel_k;
    };

    /// Push a new trajectory into the buffer.
    /// solve_timestamp MUST be the wall-clock time at which x0 was measured
    /// (i.e. solve START, not end), so that elapsed-based indexing is correct.
    void updatePlan(const std::vector<Eigen::VectorXd>& state_trajectory,
                    const std::vector<Eigen::VectorXd>& control_trajectory,
                    double /*solve_time_ms*/,
                    int solve_num,
                    bool is_relative_plan,
                    const std::chrono::steady_clock::time_point& solve_timestamp,
                    double /*ocp_dt*/,
                    int min_replay_before_swap) {
        std::lock_guard<std::mutex> lk(mutex_);
        min_replay_before_swap_ = std::max(1, min_replay_before_swap);

        // First accepted solve must become active immediately to avoid command gaps.
        if (state_trajectory_.empty()) {
            state_trajectory_ = state_trajectory;
            control_trajectory_ = control_trajectory;
            active_solve_num_ = solve_num;
            plan_is_relative_ = is_relative_plan;
            last_solve_wall_time_ = solve_timestamp;
            has_variable_dt_ = (!state_trajectory_.empty() &&
                                state_trajectory_[0].size() > IDX_DT);
            return;
        }

        // Keep only the freshest pending plan; swap-in decision happens in sample().
        pending_state_trajectory_ = state_trajectory;
        pending_control_trajectory_ = control_trajectory;
        pending_solve_num_ = solve_num;
        pending_plan_is_relative_ = is_relative_plan;
        pending_solve_wall_time_ = solve_timestamp;
        pending_has_variable_dt_ = (!pending_state_trajectory_.empty() &&
                                    pending_state_trajectory_[0].size() > IDX_DT);
        has_pending_plan_ = true;
    }

    /// Sample the active trajectory at the current wall-clock time.
    /// Returns an interpolated setpoint between the bracketing nodes.
    ReplaySample sample(const std::chrono::steady_clock::time_point& now,
                        double ocp_dt) {
        ReplaySample s;
        s.x_cmd = Eigen::VectorXd::Zero(NX_PHYS);
        s.u_cmd = Eigen::VectorXd::Zero(4);

        std::lock_guard<std::mutex> lk(mutex_);

        maybeSwapPendingPlan(now, ocp_dt);

        if (state_trajectory_.empty()) {
            return s;
        }

        s.has_plan = true;
        s.active_solve_num = active_solve_num_;
        s.plan_is_relative = plan_is_relative_;
        s.elapsed = std::chrono::duration<double>(now - last_solve_wall_time_).count();

        const int N = static_cast<int>(state_trajectory_.size()) - 1;
        if (N < 1) return s;

        // --- Compute horizon end and clamped sample time ---
        s.horizon_end = nodeTime(N, ocp_dt);
        const double t_sample = std::clamp(s.elapsed, 0.0, s.horizon_end);
        const double t_lookahead = std::clamp(t_sample + ocp_dt, 0.0, s.horizon_end);

        s.x_cmd = sampleStateAt(t_sample, ocp_dt).head(NX_PHYS);
        s.x_cmd_lookahead = sampleStateAt(t_lookahead, ocp_dt).head(NX_PHYS);
        if (plan_is_relative_) {
            s.x_rel_k = sampleStateAt(t_sample, ocp_dt);
        }
        const int k_lo = findLowerNode(t_sample, N, ocp_dt);

        // --- Control: zero-order hold from lower node, truncated to 4-dim ---
        if (!control_trajectory_.empty()) {
            const int u_idx = std::min(k_lo, static_cast<int>(control_trajectory_.size()) - 1);
            const auto& u_full = control_trajectory_[u_idx];
            s.u_cmd = u_full.head(std::min(static_cast<int>(u_full.size()), 4));
        }

        return s;
    }

private:
    static constexpr int IDX_DT = 13;
    static constexpr int NX_PHYS = 13;

    int findLowerNode(double t_sample, int N, double ocp_dt) const {
        int k_lo = 0;
        for (int k = 0; k < N; ++k) {
            if (nodeTime(k + 1, ocp_dt) > t_sample) {
                k_lo = k;
                break;
            }
            k_lo = k;
        }
        return k_lo;
    }

    Eigen::VectorXd sampleStateAt(double t_sample, double ocp_dt) const {
        const int N = static_cast<int>(state_trajectory_.size()) - 1;
        if (N < 1) {
            return Eigen::VectorXd::Zero(NX_PHYS);
        }

        const int k_lo = findLowerNode(t_sample, N, ocp_dt);
        const int k_hi = std::min(k_lo + 1, N);

        double alpha = 0.0;
        if (k_lo < k_hi) {
            const double dt_seg = nodeTime(k_hi, ocp_dt) - nodeTime(k_lo, ocp_dt);
            if (dt_seg > 1e-9) {
                alpha = std::clamp((t_sample - nodeTime(k_lo, ocp_dt)) / dt_seg, 0.0, 1.0);
            }
        }

        const auto& x_lo = state_trajectory_[k_lo];
        const auto& x_hi = state_trajectory_[k_hi];
        Eigen::VectorXd x_interp = Eigen::VectorXd::Zero(std::max((int)x_lo.size(), NX_PHYS));
        const int interp_len = std::min({NX_PHYS, (int)x_lo.size(), (int)x_hi.size()});

        if (interp_len >= 6) {
            x_interp.segment(0, 6) = (1.0 - alpha) * x_lo.head(6) + alpha * x_hi.head(6);
        }

        if (interp_len >= 10) {
            Eigen::Vector4d q_lo = x_lo.segment(6, 4);
            Eigen::Vector4d q_hi = x_hi.segment(6, 4);
            if (q_lo.dot(q_hi) < 0.0) {
                q_hi = -q_hi;
            }
            Eigen::Vector4d q_interp = (1.0 - alpha) * q_lo + alpha * q_hi;
            const double qn = q_interp.norm();
            if (qn > 1e-9) {
                q_interp /= qn;
            }
            x_interp.segment(6, 4) = q_interp;
        }

        if (interp_len >= 13) {
            x_interp.segment(10, 3) =
                (1.0 - alpha) * x_lo.segment(10, 3) + alpha * x_hi.segment(10, 3);
        }

        if (plan_is_relative_ && x_lo.size() > NX_PHYS) {
            for (int i = NX_PHYS; i < x_lo.size(); ++i) {
                x_interp(i) = x_lo(i);
            }
        }

        return x_interp;
    }

    void maybeSwapPendingPlan(const std::chrono::steady_clock::time_point& now,
                              double ocp_dt) {
        if (!has_pending_plan_) {
            return;
        }

        // Safety override: if active plan is stale, promote freshest pending immediately.
        const double elapsed = std::chrono::duration<double>(now - last_solve_wall_time_).count();
        const int active_N = static_cast<int>(state_trajectory_.size()) - 1;
        const double horizon_end = (active_N > 0) ? nodeTime(active_N, ocp_dt) : 0.0;
        const bool active_stale = (active_N <= 0) || (elapsed > horizon_end + 0.2);

        // Main gate: enforce minimum replay duration before switching plans.
        const double min_hold_time = std::max(0.0, static_cast<double>(min_replay_before_swap_ - 1) * ocp_dt);
        const bool min_replay_elapsed = (elapsed >= min_hold_time);

        if (!active_stale && !min_replay_elapsed) {
            return;
        }

        state_trajectory_ = pending_state_trajectory_;
        control_trajectory_ = pending_control_trajectory_;
        active_solve_num_ = pending_solve_num_;
        plan_is_relative_ = pending_plan_is_relative_;
        last_solve_wall_time_ = pending_solve_wall_time_;
        has_variable_dt_ = pending_has_variable_dt_;

        pending_state_trajectory_.clear();
        pending_control_trajectory_.clear();
        pending_solve_num_ = -1;
        pending_plan_is_relative_ = false;
        pending_has_variable_dt_ = false;
        pending_solve_wall_time_ = std::chrono::steady_clock::time_point{};
        has_pending_plan_ = false;
    }

    double nodeTime(int k, double ocp_dt) const {
        if (has_variable_dt_ &&
            k < static_cast<int>(state_trajectory_.size()) &&
            state_trajectory_[k].size() > IDX_DT) {
            return state_trajectory_[k](IDX_DT);
        }
        return k * ocp_dt;
    }

    std::mutex mutex_;
    std::vector<Eigen::VectorXd> state_trajectory_;
    std::vector<Eigen::VectorXd> control_trajectory_;
    int active_solve_num_ = -1;
    bool plan_is_relative_ = false;
    bool has_variable_dt_ = false;
    std::chrono::steady_clock::time_point last_solve_wall_time_{};

    std::vector<Eigen::VectorXd> pending_state_trajectory_;
    std::vector<Eigen::VectorXd> pending_control_trajectory_;
    int pending_solve_num_ = -1;
    bool pending_plan_is_relative_ = false;
    bool pending_has_variable_dt_ = false;
    std::chrono::steady_clock::time_point pending_solve_wall_time_{};
    bool has_pending_plan_ = false;
    int min_replay_before_swap_ = 1;
};
