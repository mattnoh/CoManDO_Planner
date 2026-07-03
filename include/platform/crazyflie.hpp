/// @file crazyflie.hpp  (ROS1 Noetic branch — stub)
/// @brief Crazyflie is ROS2-only. This stub provides the State struct so
///        state_monitor.hpp compiles. No publishers/subscribers/setup.
#pragma once

#include <Eigen/Dense>

namespace platform {
namespace crazyflie {

struct State {
    Eigen::VectorXd current = Eigen::VectorXd::Zero(13);
    bool pose_received = false;
    bool odom_received = false;
    bool hasFullState() const { return odom_received; }
    void reset() { pose_received = false; odom_received = false; current.setZero(); }
};

// Dummy Handles type so any lingering include of this header doesn't break.
struct Handles {
    void reset() {}
};

}  // namespace crazyflie
}  // namespace platform
