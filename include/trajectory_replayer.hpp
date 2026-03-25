/// @file trajectory_replayer.hpp
/// @brief Thread-safe replay buffer and time-indexed sampler for MPC trajectories.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <vector>

class TrajectoryReplayer {
public:
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

    void updatePlan(const std::vector<Eigen::VectorXd>& state_trajectory,
                    const std::vector<Eigen::VectorXd>& control_trajectory,
                    double solve_time_ms,
                    int solve_num,
                    bool is_relative_plan,
                    const std::chrono::steady_clock::time_point& solve_timestamp,
                    double ocp_dt) {
        std::lock_guard<std::mutex> lk(mutex_);
        state_trajectory_ = state_trajectory;
        control_trajectory_ = control_trajectory;

        const int k = static_cast<int>(std::round(solve_time_ms / (ocp_dt * 1000.0)));
        const int N = static_cast<int>(state_trajectory_.size()) - 1;
        replay_idx_ = std::min(1 + k, N);

        active_solve_num_ = solve_num;
        plan_is_relative_ = is_relative_plan;
        last_solve_wall_time_ = solve_timestamp;
    }

    ReplaySample sample(const std::chrono::steady_clock::time_point& now,
                        double ocp_dt) {
        constexpr int IDX_DT = 13;

        ReplaySample s;
        s.x_cmd = Eigen::VectorXd::Zero(13);
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
        if (!s.plan_is_relative) {
            int idx = std::min(static_cast<int>(s.elapsed / ocp_dt), N);
            idx = std::max(0, idx);
            replay_idx_ = idx;

            if (state_trajectory_[idx].size() >= 13) {
                s.x_cmd = state_trajectory_[idx].head(13);
            }
            if (!control_trajectory_.empty()) {
                const int u_idx = std::min(idx, static_cast<int>(control_trajectory_.size()) - 1);
                s.u_cmd = control_trajectory_[u_idx];
            }
            s.horizon_end = N * ocp_dt;
            return s;
        }

        int idx = 0;
        while (idx + 1 < static_cast<int>(state_trajectory_.size()) &&
               state_trajectory_[idx].size() > IDX_DT &&
               state_trajectory_[idx](IDX_DT) < s.elapsed) {
            ++idx;
        }
        idx = std::min(idx, N);
        replay_idx_ = idx;

        s.x_rel_k = state_trajectory_[idx];
        if (!control_trajectory_.empty()) {
            const int u_idx = std::min(idx, static_cast<int>(control_trajectory_.size()) - 1);
            s.u_cmd = control_trajectory_[u_idx];
        }

        if (!state_trajectory_.empty() && state_trajectory_.back().size() > IDX_DT) {
            s.horizon_end = state_trajectory_.back()(IDX_DT);
        }
        return s;
    }

private:
    std::mutex mutex_;
    std::vector<Eigen::VectorXd> state_trajectory_;
    std::vector<Eigen::VectorXd> control_trajectory_;
    int replay_idx_ = 1;
    int active_solve_num_ = -1;
    bool plan_is_relative_ = false;
    std::chrono::steady_clock::time_point last_solve_wall_time_{};
};
