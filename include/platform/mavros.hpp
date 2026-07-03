/// @file mavros.hpp  (ROS1 Noetic branch)
/// @brief MAVROS platform adapter — roscpp version.
///
/// State input:  /mavros/local_position/odom (nav_msgs/Odometry) → 13D ENU
/// FCU state:    /mavros/state (mavros_msgs/State) → OFFBOARD/armed truth
/// CmdBodyRate:  /mavros/setpoint_raw/attitude (mavros_msgs/AttitudeTarget)
///               type_mask=IGNORE_ATTITUDE(128); body_rate; thrust=T*hover_thrust/9.81
/// CmdFullState: /mavros/setpoint_raw/local (mavros_msgs/PositionTarget)
///               position + velocity + yaw, ENU data (mavros converts ENU→NED)
///
/// Startup sequence (heartbeat-driven, 10 Hz):
///   WarmingUp: stream position-hold setpoints for 2 s (PX4 requires a >2 Hz
///              setpoint stream before it will accept OFFBOARD), then
///   ReadyToArm: request OFFBOARD, then ARM, each retried at <=1 Hz until
///              /mavros/state confirms them, then
///   Armed:     confirmed by /mavros/state (mode==OFFBOARD && armed).
///
/// The heartbeat always re-streams the last command (body-rate or full-state)
/// or a position hold when no command has been issued yet, so PX4 never
/// starves of setpoints between OFFBOARD engagement and the first solve, or
/// while the planner is paused.
///
/// Safety: OFFBOARD/ARM requests are only retried during the INITIAL
/// engagement. Once the FCU has been armed in OFFBOARD, a later disarm or
/// mode change (pilot takeover, failsafe, auto-disarm after landing) is
/// logged but never fought with re-requests.
#pragma once

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/CommandLong.h>
#include <mavros_msgs/State.h>
#include <Eigen/Dense>
#include <mutex>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace platform {
namespace mavros {

enum class ArmState { Idle, WarmingUp, ReadyToArm, Armed };

struct Handles {
    Eigen::VectorXd current = Eigen::VectorXd::Zero(13);
    bool odom_received = false;

    ArmState arm_state = ArmState::Idle;
    std::chrono::steady_clock::time_point warmup_start;
    float hover_thrust = 0.3f;

    // Latest FCU truth from /mavros/state (guarded: written by the state
    // callback, read by the heartbeat; AsyncSpinner runs them concurrently).
    std::mutex fcu_mutex;
    bool fcu_connected = false;
    bool fcu_armed = false;
    std::string fcu_mode;
    bool fcu_state_received = false;
    bool ever_armed = false;

    ros::Time last_mode_request{0.0};
    ros::Time last_arm_request{0.0};

    Eigen::VectorXd last_state_cmd = Eigen::VectorXd::Zero(13);
    Eigen::VectorXd last_rate_cmd  = Eigen::VectorXd::Zero(5);
    bool last_cmd_is_bodyrate = false;
    bool has_state_cmd = false;
    bool has_rate_cmd = false;

    // Touchdown handoff: once engaged, the planner stops commanding and the
    // heartbeat drives PX4 to a stop ("auto_land" → AUTO.LAND + PX4
    // auto-disarm, "disarm" → direct disarm request).
    bool land_engaged = false;
    bool land_complete_logged = false;
    std::string land_action;
    ros::Time last_land_request{0.0};

    ros::Subscriber    odom_sub;
    ros::Subscriber    state_sub;
    ros::Publisher     att_pub;
    ros::Publisher     pos_pub;
    ros::ServiceClient set_mode_client;
    ros::ServiceClient arming_client;
    ros::ServiceClient command_client;   // CommandLong (force disarm)
    ros::Timer         heartbeat_timer;
};

inline mavros_msgs::AttitudeTarget neutralAttitudeTarget() {
    mavros_msgs::AttitudeTarget msg;
    msg.header.stamp = ros::Time::now();
    msg.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE;
    msg.body_rate.x = msg.body_rate.y = msg.body_rate.z = 0.0f;
    msg.thrust = 0.0f;
    return msg;
}

inline void publishAttitudeTargetMsg(Handles& h, const Eigen::VectorXd& u) {
    if (h.land_engaged) return;  // control handed to PX4 for touchdown/stop
    mavros_msgs::AttitudeTarget msg;
    msg.header.stamp = ros::Time::now();
    msg.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE;
    msg.body_rate.x = u.size() > 1 ? float(u(1)) : 0.0f;
    msg.body_rate.y = u.size() > 2 ? float(u(2)) : 0.0f;
    msg.body_rate.z = u.size() > 3 ? float(u(3)) : 0.0f;
    msg.thrust = u.size() > 0
        ? float(std::clamp(u(0) * h.hover_thrust / 9.81, 0.0, 1.0))
        : 0.0f;
    h.att_pub.publish(msg);
}

inline void publishPositionTargetMsg(Handles& h, const Eigen::VectorXd& s) {
    if (h.land_engaged) return;  // control handed to PX4 for touchdown/stop
    mavros_msgs::PositionTarget msg;
    msg.header.stamp = ros::Time::now();
    msg.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    msg.type_mask = mavros_msgs::PositionTarget::IGNORE_AFX |
                    mavros_msgs::PositionTarget::IGNORE_AFY |
                    mavros_msgs::PositionTarget::IGNORE_AFZ |
                    mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
    if (s.size() >= 3) { msg.position.x=float(s(0)); msg.position.y=float(s(1)); msg.position.z=float(s(2)); }
    if (s.size() >= 6) { msg.velocity.x=float(s(3)); msg.velocity.y=float(s(4)); msg.velocity.z=float(s(5)); }
    if (s.size() >= 10) {
        const double qw=s(6), qx=s(7), qy=s(8), qz=s(9);
        msg.yaw = float(std::atan2(2.0*(qw*qz+qx*qy), 1.0-2.0*(qy*qy+qz*qz)));
    }
    h.pos_pub.publish(msg);
}

/// Position hold at the current odom pose (zero velocity, current yaw).
/// Falls back to a neutral AttitudeTarget until the first odom arrives.
inline void publishHoldPositionTarget(Handles& h) {
    if (!h.odom_received) {
        h.att_pub.publish(neutralAttitudeTarget());
        return;
    }
    Eigen::VectorXd hold = h.current;
    if (hold.size() >= 6) hold.segment(3, 3).setZero();
    publishPositionTargetMsg(h, hold);
}

/// Drive the touchdown handoff until PX4 reports disarmed.
inline void landActionTick(Handles& h) {
    bool fcu_seen, armed;
    std::string mode;
    {
        std::lock_guard<std::mutex> lk(h.fcu_mutex);
        fcu_seen = h.fcu_state_received;
        armed = h.fcu_armed;
        mode = h.fcu_mode;
    }
    if (!fcu_seen) return;
    if (!armed) {
        if (!h.land_complete_logged) {
            h.land_complete_logged = true;
            ROS_INFO("[MAVROS] Disarmed — landing complete (action '%s')",
                     h.land_action.c_str());
        }
        return;
    }
    const ros::Time now = ros::Time::now();
    if ((now - h.last_land_request).toSec() < 1.0) return;
    h.last_land_request = now;
    if (h.land_action == "auto_land") {
        if (mode != "AUTO.LAND") {
            mavros_msgs::SetMode mode_req;
            mode_req.request.custom_mode = "AUTO.LAND";
            if (h.set_mode_client.call(mode_req))
                ROS_INFO_THROTTLE(5.0, "[MAVROS] AUTO.LAND requested (current mode=%s)", mode.c_str());
            else
                ROS_WARN_THROTTLE(5.0, "[MAVROS] AUTO.LAND SetMode call failed");
        }
        // PX4 auto-disarms after landing (COM_DISARM_LAND).
    } else if (h.land_action == "force_disarm") {
        // Immediate motor cut (MAV_CMD_COMPONENT_ARM_DISARM, param2=21196
        // bypasses the land detector). For moving-platform touchdowns: the
        // planner only engages this at terminal freeze, i.e. on the deck.
        mavros_msgs::CommandLong cmd;
        cmd.request.command = 400;   // MAV_CMD_COMPONENT_ARM_DISARM
        cmd.request.param1 = 0.0;    // disarm
        cmd.request.param2 = 21196;  // force
        if (h.command_client.call(cmd))
            ROS_INFO_THROTTLE(5.0, "[MAVROS] Force-disarm requested");
        else
            ROS_WARN_THROTTLE(5.0, "[MAVROS] Force-disarm CommandLong call failed");
    } else {  // "disarm": PX4 accepts once its land detector agrees
        mavros_msgs::CommandBool arm_req;
        arm_req.request.value = false;
        if (h.arming_client.call(arm_req))
            ROS_INFO_THROTTLE(5.0, "[MAVROS] Disarm requested");
        else
            ROS_WARN_THROTTLE(5.0, "[MAVROS] Disarm service call failed");
    }
}

/// Called by the planner at touchdown (terminal freeze on a landing OCP).
/// Stops all offboard setpoints and hands the stop to PX4.
inline void engageLandAction(Handles& h, const std::string& action) {
    if (h.land_engaged) return;
    h.land_action = action;
    h.land_engaged = true;
    h.land_complete_logged = false;
    h.last_land_request = ros::Time(0.0);
    ROS_INFO("[MAVROS] Touchdown — engaging land action '%s', offboard setpoints stopped", action.c_str());
}

/// Called when a new command_seq is accepted. If the vehicle is on the ground
/// (disarmed after a completed landing), clear the one-shot engagement guards
/// so the heartbeat re-engages OFFBOARD+ARM for the new flight.
inline void resetForNewFlight(Handles& h) {
    bool armed;
    {
        std::lock_guard<std::mutex> lk(h.fcu_mutex);
        armed = h.fcu_armed;
    }
    if (armed) return;  // in-flight command change: keep the safety guards
    h.land_engaged = false;
    h.land_complete_logged = false;
    h.ever_armed = false;
    h.has_state_cmd = false;   // stream position-hold at current pose pre-arm,
    h.has_rate_cmd = false;    // not the previous flight's last command
    h.last_mode_request = ros::Time(0.0);
    h.last_arm_request = ros::Time(0.0);
    if (h.arm_state == ArmState::Armed) h.arm_state = ArmState::ReadyToArm;
    ROS_INFO("[MAVROS] New command while disarmed — re-engaging OFFBOARD+ARM");
}

inline void heartbeatTick(Handles& h) {
    using namespace std::chrono;

    // 0) After touchdown the planner no longer commands; drive PX4 to a stop.
    if (h.land_engaged) {
        landActionTick(h);
        return;
    }

    // 1) Keepalive: always stream a setpoint. PX4 refuses OFFBOARD without a
    //    >2 Hz stream and fails over out of it when the stream stops.
    if (h.has_rate_cmd && h.last_cmd_is_bodyrate) {
        publishAttitudeTargetMsg(h, h.last_rate_cmd);
    } else if (h.has_state_cmd) {
        publishPositionTargetMsg(h, h.last_state_cmd);
    } else {
        publishHoldPositionTarget(h);
    }

    // 2) Warmup: stream-only phase before any OFFBOARD/ARM request.
    if (h.arm_state == ArmState::WarmingUp) {
        const double elapsed =
            duration<double>(steady_clock::now() - h.warmup_start).count();
        if (elapsed < 2.0) return;
        h.arm_state = ArmState::ReadyToArm;
        ROS_INFO("[MAVROS] Warmup done, engaging OFFBOARD+ARM (retried until /mavros/state confirms)");
    }
    if (h.arm_state == ArmState::Idle) return;

    bool fcu_seen, armed;
    std::string mode;
    {
        std::lock_guard<std::mutex> lk(h.fcu_mutex);
        fcu_seen = h.fcu_state_received;
        armed = h.fcu_armed;
        mode = h.fcu_mode;
    }
    if (!fcu_seen) {
        ROS_WARN_THROTTLE(5.0, "[MAVROS] No /mavros/state yet — is mavros connected to the FCU?");
        return;
    }

    // 3) Drive arm_state from FCU truth.
    const bool offboard = (mode == "OFFBOARD");
    if (offboard && armed) {
        if (h.arm_state != ArmState::Armed) {
            h.arm_state = ArmState::Armed;
            h.ever_armed = true;
            ROS_INFO("[MAVROS] OFFBOARD + armed confirmed by /mavros/state");
        }
        return;
    }
    if (h.arm_state == ArmState::Armed) {
        h.arm_state = ArmState::ReadyToArm;
        ROS_WARN("[MAVROS] FCU left OFFBOARD/armed (mode=%s armed=%d) — not re-requesting after initial engagement",
                 mode.c_str(), armed ? 1 : 0);
    }

    // 4) Initial engagement retries only (see safety note in the header).
    if (h.ever_armed) return;
    if (!h.odom_received) return;

    const ros::Time now = ros::Time::now();
    if (!offboard) {
        if ((now - h.last_mode_request).toSec() >= 1.0) {
            h.last_mode_request = now;
            mavros_msgs::SetMode mode_req;
            mode_req.request.custom_mode = "OFFBOARD";
            if (h.set_mode_client.call(mode_req))
                ROS_INFO_THROTTLE(5.0, "[MAVROS] OFFBOARD mode requested (current mode=%s)", mode.c_str());
            else
                ROS_WARN_THROTTLE(5.0, "[MAVROS] SetMode service call failed");
        }
    } else if (!armed) {
        if ((now - h.last_arm_request).toSec() >= 1.0) {
            h.last_arm_request = now;
            mavros_msgs::CommandBool arm_req;
            arm_req.request.value = true;
            if (h.arming_client.call(arm_req))
                ROS_INFO_THROTTLE(5.0, "[MAVROS] ARM requested");
            else
                ROS_WARN_THROTTLE(5.0, "[MAVROS] Arming service call failed");
        }
    }
}

inline void setup(ros::NodeHandle& nh,
                  float hover_thrust_param,
                  Eigen::VectorXd& state_out,
                  std::mutex& state_mutex,
                  Handles& h)
{
    h.hover_thrust = hover_thrust_param;

    h.odom_sub = nh.subscribe<nav_msgs::Odometry>(
        "/mavros/local_position/odom", 10,
        [&state_out, &state_mutex, &h](const nav_msgs::Odometry::ConstPtr& msg) {
            std::lock_guard<std::mutex> lk(state_mutex);
            const auto& p = msg->pose.pose.position;
            const auto& q = msg->pose.pose.orientation;
            const auto& v = msg->twist.twist.linear;
            const auto& w = msg->twist.twist.angular;
            state_out(0)=p.x; state_out(1)=p.y; state_out(2)=p.z;
            state_out(3)=v.x; state_out(4)=v.y; state_out(5)=v.z;
            state_out(6)=q.w; state_out(7)=q.x;
            state_out(8)=q.y; state_out(9)=q.z;
            state_out(10)=w.x; state_out(11)=w.y; state_out(12)=w.z;
            h.current = state_out;
            h.odom_received = true;
        });

    h.state_sub = nh.subscribe<mavros_msgs::State>(
        "/mavros/state", 10,
        [&h](const mavros_msgs::State::ConstPtr& msg) {
            std::lock_guard<std::mutex> lk(h.fcu_mutex);
            h.fcu_connected = msg->connected;
            h.fcu_armed = msg->armed;
            h.fcu_mode = msg->mode;
            h.fcu_state_received = true;
        });

    h.att_pub = nh.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude", 10);
    h.pos_pub = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 10);
    h.set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
    h.arming_client   = nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    h.command_client  = nh.serviceClient<mavros_msgs::CommandLong>("/mavros/cmd/command");

    h.arm_state    = ArmState::WarmingUp;
    h.warmup_start = std::chrono::steady_clock::now();
    ROS_INFO("[MAVROS] Warmup started (2 s of hold setpoints before OFFBOARD+ARM)");

    h.heartbeat_timer = nh.createTimer(
        ros::Duration(0.1),
        [&h](const ros::TimerEvent&) { heartbeatTick(h); });

    ROS_INFO("[MAVROS] Platform setup done. hover_thrust=%.3f", hover_thrust_param);
}

inline void publishBodyRateCommand(Handles& h, const Eigen::VectorXd& u) {
    h.last_rate_cmd = u;
    h.last_cmd_is_bodyrate = true;
    h.has_rate_cmd = true;
    publishAttitudeTargetMsg(h, u);
}

inline void publishFullStateCommand(Handles& h, const Eigen::VectorXd& s) {
    h.last_state_cmd = s;
    h.last_cmd_is_bodyrate = false;
    h.has_state_cmd = true;
    publishPositionTargetMsg(h, s);
}

}  // namespace mavros
}  // namespace platform
