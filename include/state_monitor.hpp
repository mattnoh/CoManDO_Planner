/// @file state_monitor.hpp
/// @brief Thread-safe state ownership for planner sensor and target streams.

#pragma once

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include <mutex>
#include <string>
#include <utility>

#include "platform/crazyflie.hpp"
#include "platform/px4.hpp"
#include "platform/target_tracker.hpp"
#include "ocp_registry.hpp"

class StateMonitor {
public:
    explicit StateMonitor(double target_state_max_age_sec = 0.2)
        : target_state_max_age_sec_(target_state_max_age_sec) {
        fallback_state_ = Eigen::VectorXd::Zero(13);
        fallback_state_(6) = 1.0;
    }

    platform::crazyflie::State& crazyflieState() { return cf_state_; }
    platform::px4::State& px4State() { return px4_state_; }
    platform::target_tracker::TargetState& targetState() { return target_state_; }
    std::mutex& stateMutex() { return state_mutex_; }
    std::mutex& targetMutex() { return target_mutex_; }

    bool hasState(const std::string& platform) const {
        if (platform == "crazyflie") {
            return cf_state_.hasFullState();
        }
        if (platform == "px4") {
            return px4_state_.hasFullState();
        }
        return false;
    }

    /// @brief Check if target state is fresh if the OCP requires it.
    bool hasFreshTargetState(bool needs_target, const rclcpp::Time& now) const {
        if (!needs_target) {
            return true;
        }
        std::lock_guard<std::mutex> lk(target_mutex_);
        return target_state_.isFresh(now, target_state_max_age_sec_);
    }

    /// @brief Get live target snapshot if the OCP requires it.
    TargetSnapshot getTargetSnapshot(bool needs_target, const rclcpp::Time& now) const {
        TargetSnapshot s;
        if (!needs_target) {
            s.valid = true;
            return s;
        }

        std::lock_guard<std::mutex> lk(target_mutex_);
        s.position = target_state_.position;
        s.velocity = target_state_.velocity;
        s.acceleration = target_state_.acceleration;
        s.odom_timestamp = target_state_.odom_timestamp;
        s.accel_timestamp = target_state_.accel_timestamp;
        s.valid = target_state_.isFresh(now, target_state_max_age_sec_);
        return s;
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d> getTargetPositionVelocity() const {
        std::lock_guard<std::mutex> lk(target_mutex_);
        return {target_state_.position, target_state_.velocity};
    }

    bool hasTargetTrajectory(const rclcpp::Time& now) const {
        std::lock_guard<std::mutex> lk(target_mutex_);
        return target_state_.predicted_trajectory.isFresh(now, 2.0);
    }

    target_models::TargetAccelBuffer getTargetAccelBuffer() const {
        std::lock_guard<std::mutex> lk(target_mutex_);
        return target_state_.predicted_trajectory.accel_buffer;
    }

    Eigen::VectorXd getCurrentState(const std::string& platform) const {
        std::lock_guard<std::mutex> lk(state_mutex_);
        if (platform == "crazyflie") {
            return cf_state_.current;
        }
        if (platform == "px4") {
            return px4_state_.current;
        }
        return fallback_state_;
    }

private:
    platform::crazyflie::State cf_state_;
    platform::px4::State px4_state_;
    platform::target_tracker::TargetState target_state_;

    mutable std::mutex state_mutex_;
    mutable std::mutex target_mutex_;
    double target_state_max_age_sec_ = 0.2;
    Eigen::VectorXd fallback_state_;
};
