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

struct TargetSnapshot {
    bool valid = false;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    rclcpp::Time odom_timestamp{0, 0, RCL_ROS_TIME};
    rclcpp::Time accel_timestamp{0, 0, RCL_ROS_TIME};
};

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

    /// @brief Check if target state is fresh for the given OCP type.
    /// For stateswitch and tracking_circle, target state must be fresh.
    bool hasFreshTargetState(const std::string& ocp_type,
                              const rclcpp::Time& now) const {
        if (ocp_type != "stateswitch" && ocp_type != "tracking_circle") {
            return true;
        }
        std::lock_guard<std::mutex> lk(target_mutex_);
        return target_state_.isFresh(now, target_state_max_age_sec_);
    }

    /// @brief Get target snapshot for the given OCP type.
    /// For stateswitch and tracking_circle, returns live target state.
    TargetSnapshot getTargetSnapshot(const std::string& ocp_type,
                                      const rclcpp::Time& now) const {
        TargetSnapshot s;
        if (ocp_type != "stateswitch" && ocp_type != "tracking_circle") {
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
