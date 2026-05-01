/// @file planner_core/command_adapter.hpp
/// @brief ROS-free OCP-output to platform-command conversion.
#pragma once

#include <Eigen/Dense>

#include <algorithm>

#include "planner_core/ocp_descriptor.hpp"
#include "dynamics/quad_6dof_target_frame.h"
#include "planner_core/types.hpp"

namespace planner_core {

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
        const Eigen::Matrix3d R_NB = Quad6DOFVarTime<double>::calcC(q_NB);
        const Eigen::Vector3d v_body = R_NB.transpose() * state.segment(3, 3);
        vx_body = v_body.x();
        vy_body = v_body.y();
        z_world = target.position.z() + state(2);
    } else if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative &&
               state.size() >= 10) {
        vx_body = state(3);
        vy_body = state(4);
        const Eigen::Vector4d q_NB = state.segment(6, 4);
        const Eigen::Matrix3d R_WN = Quad6DOFVarTime<double>::calcC(target.orientation);
        const Eigen::Matrix3d R_WB = R_WN * Quad6DOFVarTime<double>::calcC(q_NB);
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
