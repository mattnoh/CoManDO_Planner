/// @file planner_core/adapters.hpp
/// @brief ROS-free frame validation and platform-command conversion.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <string>

#include "dynamics/quad_6dof_target_frame.h"
#include "planner_core/types.hpp"

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

inline Eigen::Vector4d makeCrazyflieHoverCommand(const OCPDescriptor& desc,
                                                 const Eigen::VectorXd& state,
                                                 const Eigen::VectorXd& control,
                                                 const TargetSnapshot& target) {
    Eigen::Vector4d hover = Eigen::Vector4d::Zero();
    if (state.size() < 6) {
        return hover;
    }

    double vx_body = 0.0;
    double vy_body = 0.0;
    double z_world = (state.size() > 2) ? state(2) : target.position.z();

    if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative &&
        state.size() >= 10) {
        const Eigen::Vector4d q_NB = state.segment(6, 4);
        const Eigen::Matrix3d R_WN = Quad6DOFVarTime<double>::calcC(target.orientation);
        const Eigen::Matrix3d R_NB = Quad6DOFVarTime<double>::calcC(q_NB);
        const Eigen::Matrix3d R_WB = R_WN * R_NB;
        const Eigen::Vector3d p_N = state.segment(0, 3);
        const Eigen::Vector3d v_N = state.segment(3, 3);
        const Eigen::Vector3d v_world =
            target.velocity + R_WN * (v_N + target.angular_velocity.cross(p_N));
        const Eigen::Vector3d v_body = R_WB.transpose() * v_world;
        vx_body = v_body.x();
        vy_body = v_body.y();
        z_world = target.position.z() + (R_WN * p_N).z();
    } else if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative &&
               state.size() >= 10) {
        const Eigen::Vector4d q_NB = state.segment(6, 4);
        const Eigen::Matrix3d R_WN = Quad6DOFVarTime<double>::calcC(target.orientation);
        const Eigen::Matrix3d R_WB = R_WN * Quad6DOFVarTime<double>::calcC(q_NB);
        const Eigen::Vector3d v_body = state.segment(3, 3) + R_WB.transpose() * target.velocity;
        vx_body = v_body.x();
        vy_body = v_body.y();
        const Eigen::Vector3d p_target_from_drone_world = R_WB * state.segment(0, 3);
        z_world = target.position.z() - p_target_from_drone_world.z();
    } else {
        vx_body = state(3);
        vy_body = state(4);
        z_world = state(2);
    }

    hover(0) = std::clamp(vx_body, -1.0, 1.0);
    hover(1) = std::clamp(vy_body, -1.0, 1.0);
    hover(2) = std::clamp(z_world, 0.1, 3.0);
    hover(3) = (control.size() > 3) ? control(3) : 0.0;
    return hover;
}

inline PlannerCommand makePlannerCommand(Platform platform,
                                         const OCPDescriptor& desc,
                                         const Eigen::VectorXd& state,
                                         const Eigen::VectorXd& control,
                                         const TargetSnapshot& target) {
    PlannerCommand cmd;
    cmd.state = state;
    cmd.control = control;

    if (desc.command_mode == OCPDescriptor::CommandMode::CmdBodyRate) {
        if (platform == Platform::Crazyflie) {
            cmd.kind = CommandKind::Hover;
            cmd.hover = makeCrazyflieHoverCommand(desc, state, control, target);
        } else {
            cmd.kind = CommandKind::BodyRate;
        }
        return cmd;
    }

    cmd.kind = CommandKind::FullState;
    if (desc.reconstruct_world_state) {
        cmd.state = desc.reconstruct_world_state(state, target);
    }
    return cmd;
}

}  // namespace planner_core
