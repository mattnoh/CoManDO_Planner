/// @file planner_core/frame_adapter.hpp
/// @brief ROS-free frame validation and OCP-state preparation.
#pragma once

#include <Eigen/Dense>

#include <string>

#include "planner_core/ocp_descriptor.hpp"

namespace planner_core {

struct FrameCheck {
    bool ok = true;
    std::string reason;
};

inline const char* frameLabel(OCPDescriptor::DroneOdomMode mode) {
    switch (mode) {
        case OCPDescriptor::DroneOdomMode::TargetFrameRelative:
            return "target_frame_relative";
        case OCPDescriptor::DroneOdomMode::BodyFrameRelative:
            return "body_relative";
        case OCPDescriptor::DroneOdomMode::AbsoluteShiftedTarget:
            return "absolute_shifted";
        case OCPDescriptor::DroneOdomMode::Absolute:
        default:
            return "absolute";
    }
}

inline FrameCheck validateFrameContract(const OCPDescriptor& desc,
                                        OCPDescriptor::DroneOdomMode active_mode,
                                        const std::string& active_odom_topic,
                                        const std::string& body_relative_topic,
                                        const std::string& target_frame_topic) {
    if (active_mode != desc.drone_odom_mode) {
        return {false, "active odom mode does not match OCP descriptor"};
    }
    if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative &&
        active_odom_topic != body_relative_topic) {
        return {false, "body-frame OCP is not subscribed to body-relative odometry"};
    }
    if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative &&
        active_odom_topic != target_frame_topic) {
        return {false, "target-frame OCP is not subscribed to target-frame odometry"};
    }
    if ((desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::Absolute ||
         desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::AbsoluteShiftedTarget) &&
        !active_odom_topic.empty()) {
        return {false, "absolute OCP has a relative odometry override active"};
    }
    return {};
}

inline Eigen::VectorXd prepareOcpState(const OCPDescriptor& desc,
                                       const Eigen::VectorXd& sensor_state,
                                       const TargetSnapshot& target) {
    if (desc.transform_state) {
        return desc.transform_state(sensor_state, target);
    }
    return sensor_state;
}

}  // namespace planner_core
