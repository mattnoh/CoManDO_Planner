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
#include "core/ocp_registry.hpp"

class StateMonitor {
public:
    struct StateDebugSnapshot {
        bool has_state = false;
        bool cf_pose_received = false;
        bool cf_odom_received = false;
        bool px4_pose_received = false;
        bool px4_odom_received = false;
    };

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

    StateDebugSnapshot getStateDebugSnapshot(const std::string& platform) const {
        std::lock_guard<std::mutex> lk(state_mutex_);
        StateDebugSnapshot snap;
        snap.cf_pose_received = cf_state_.pose_received;
        snap.cf_odom_received = cf_state_.odom_received;
        snap.px4_pose_received = px4_state_.hasFullState();
        snap.px4_odom_received = px4_state_.hasFullState();

        if (platform == "crazyflie") {
            snap.has_state = cf_state_.hasFullState();
        } else if (platform == "px4") {
            snap.has_state = px4_state_.hasFullState();
        }
        return snap;
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
        std::lock_guard<std::mutex> lk(target_mutex_);
        s.position = target_state_.position;
        s.velocity = target_state_.velocity;
        s.acceleration = target_state_.acceleration;
        s.angular_velocity = target_state_.angular_velocity;
        s.angular_acceleration = target_state_.angular_acceleration;
        s.orientation = target_state_.orientation;
        s.odom_stamp_sec = target_state_.odom_timestamp.seconds();
        s.accel_stamp_sec = target_state_.accel_timestamp.seconds();
        s.valid = target_state_.isFresh(now, target_state_max_age_sec_);

        if (!needs_target) {
            // Optional target: expose best-effort snapshot, but do not gate planner loop.
            return s;
        }
        return s;
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d> getTargetPositionVelocity() const {
        std::lock_guard<std::mutex> lk(target_mutex_);
        return {target_state_.position, target_state_.velocity};
    }

    bool hasTargetTrajectory(const rclcpp::Time& now) const {
        std::lock_guard<std::mutex> lk(target_mutex_);
        return target_state_.predicted_accel.isFresh(now, 2.0);
    }

    target_models::TargetAccelBuffer getTargetAccelBuffer() const {
        std::lock_guard<std::mutex> lk(target_mutex_);
        return target_state_.predicted_accel.accel_buffer;
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
