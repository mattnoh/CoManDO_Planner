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
    }

    struct ReplaySample {
        bool has_plan = false;
        int active_solve_num = -1;
        bool plan_is_relative = false;
        double elapsed = 0.0;
        double horizon_end = 0.0;
        Eigen::VectorXd x_cmd;
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
                    double /*ocp_dt*/) {
        std::lock_guard<std::mutex> lk(mutex_);
        state_trajectory_ = state_trajectory;
        control_trajectory_ = control_trajectory;
        active_solve_num_ = solve_num;
        plan_is_relative_ = is_relative_plan;
        last_solve_wall_time_ = solve_timestamp;

        has_variable_dt_ = (!state_trajectory_.empty() &&
                            state_trajectory_[0].size() > IDX_DT);
    }

    /// Sample the active trajectory at the current wall-clock time.
    /// Returns an interpolated setpoint between the bracketing nodes.
    ReplaySample sample(const std::chrono::steady_clock::time_point& now,
                        double ocp_dt) {
        ReplaySample s;
        s.x_cmd = Eigen::VectorXd::Zero(NX_PHYS);
        s.u_cmd = Eigen::VectorXd::Zero(4);

        std::lock_guard<std::mutex> lk(mutex_);
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

        // --- Find bracketing nodes: nodeTime(k_lo) <= t_sample < nodeTime(k_hi) ---
        int k_lo = 0;
        for (int k = 0; k < N; ++k) {
            if (nodeTime(k + 1, ocp_dt) > t_sample) { k_lo = k; break; }
            k_lo = k;
        }
        const int k_hi = std::min(k_lo + 1, N);

        // --- Interpolation factor ---
        double alpha = 0.0;
        if (k_lo < k_hi) {
            const double dt_seg = nodeTime(k_hi, ocp_dt) - nodeTime(k_lo, ocp_dt);
            if (dt_seg > 1e-9) {
                alpha = std::clamp((t_sample - nodeTime(k_lo, ocp_dt)) / dt_seg, 0.0, 1.0);
            }
        }

        // --- Interpolate physical state (13-dim) ---
        const auto& x_lo = state_trajectory_[k_lo];
        const auto& x_hi = state_trajectory_[k_hi];
        Eigen::VectorXd x_interp = Eigen::VectorXd::Zero(std::max((int)x_lo.size(), NX_PHYS));
        
        // Base interpolation for all available dims up to minimum size
        const int interp_len = std::min({NX_PHYS, (int)x_lo.size(), (int)x_hi.size()});
        
        // Position + velocity (0..5): linear
        if (interp_len >= 6) {
            x_interp.segment(0, 6) = (1.0 - alpha) * x_lo.head(6) + alpha * x_hi.head(6);
        }

        // Quaternion (6..9): LERP + normalize (exact for small angles)
        if (interp_len >= 10) {
            Eigen::Vector4d q_lo = x_lo.segment(6, 4);
            Eigen::Vector4d q_hi = x_hi.segment(6, 4);
            if (q_lo.dot(q_hi) < 0.0) q_hi = -q_hi;  // shortest path
            Eigen::Vector4d q_interp = (1.0 - alpha) * q_lo + alpha * q_hi;
            double qn = q_interp.norm();
            if (qn > 1e-9) q_interp /= qn;
            x_interp.segment(6, 4) = q_interp;
        }

        // Angular velocity (10..12): linear
        if (interp_len >= 13) {
            x_interp.segment(10, 3) = (1.0 - alpha) * x_lo.segment(10, 3) + alpha * x_hi.segment(10, 3);
        }
        
        // If relative plan, and there are more variables (e.g. IDX_DT), copy them over from x_lo 
        // so that stateswitch gets the full 14-dim relative state if needed
        if (plan_is_relative_ && x_lo.size() > NX_PHYS) {
             for (int i = NX_PHYS; i < x_lo.size(); ++i) {
                 x_interp(i) = x_lo(i);
             }
        }

        s.x_cmd = x_interp.head(NX_PHYS); // ALWAYS publish 13-dim

        if (plan_is_relative_) {
            s.x_rel_k = x_interp; // pass full relative state so x[13] is visible if needed
        }

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
};
