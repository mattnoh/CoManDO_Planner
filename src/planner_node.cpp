/// @file planner_node.cpp
/// @brief CoManDO planner node - platform-agnostic MPC planner.

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <Eigen/Dense>

#include "planner_core/quadrotor_mpc.hpp"
#include "planner_core/ocp_registry.hpp"
#include "platform/crazyflie.hpp"
#include "platform/mavros.hpp"
#include "platform/target_tracker.hpp"
#include "platform/state_monitor.hpp"
#include "trajectory_replayer.hpp"
#include "planner_runtime_config.hpp"
#include "planner_logging.hpp"
#include "planner_core/adapters.hpp"
#include "planner_core/planner_core.hpp"
#include "hover_controller.hpp"

#include <chrono>
#include <memory>
#include <cmath>
#include <mutex>
#include <atomic>
#include <vector>
#include <deque>
#include <algorithm>
#include <sstream>
#include <array>
#include <limits>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("comando_planner") {
        PlannerRuntimeConfig runtime_cfg = loadPlannerRuntimeConfig(this);
        runtime_cfg.t_start_abs = this->now().seconds();
        runtime_cfg_ = runtime_cfg;

        ocp_type_ = runtime_cfg.ocp_type;
        drone_name_ = runtime_cfg.drone_name;
        logging_enabled_ = runtime_cfg.enable_logging;
        platform_ = runtime_cfg.platform;
        solver_type_ = runtime_cfg.solver;
        mode_ = runtime_cfg.mode;
        n_replay_ = runtime_cfg.n_replay;
        mass_kg_ = runtime_cfg.mass_kg;
        // hover_thrust: normalized [0,1] throttle at hover — used by MAVROS platform.
        // Read from the flight stack hover-thrust estimate after calibration:
        //   ros2 service call /mavros/param/get mavros_msgs/srv/ParamGet "{param_id: MPC_THR_HOVER}"
        this->declare_parameter("hover_thrust", 0.3);
        hover_thrust_param_ = float(this->get_parameter("hover_thrust").as_double());
        this->declare_parameter("skip_trajectory_validation", false);
        skip_trajectory_validation_ = this->get_parameter("skip_trajectory_validation").as_bool();
        this->declare_parameter("max_first_solve_age_sec", 1.0);
        max_first_solve_age_sec_ = this->get_parameter("max_first_solve_age_sec").as_double();
        this->declare_parameter("max_relative_position_norm", 10.0);
        max_relative_position_norm_ = this->get_parameter("max_relative_position_norm").as_double();
        this->declare_parameter("max_relative_vertical_abs", 5.0);
        max_relative_vertical_abs_ = this->get_parameter("max_relative_vertical_abs").as_double();
        this->declare_parameter("max_relative_velocity_norm", 8.0);
        max_relative_velocity_norm_ = this->get_parameter("max_relative_velocity_norm").as_double();
        open_loop_abort_on_divergence_ = runtime_cfg.open_loop_abort_on_divergence;
        open_loop_abort_max_z_error_m_ = runtime_cfg.open_loop_abort_max_z_error_m;
        open_loop_abort_max_vz_error_mps_ = runtime_cfg.open_loop_abort_max_vz_error_mps;
        target_odom_topic_ = runtime_cfg.target_odom_topic;
        target_accel_topic_ = runtime_cfg.target_accel_topic;
        target_predicted_accel_topic_ = runtime_cfg.target_predicted_accel_topic;
        debug_body_relative_trace_ = runtime_cfg.debug_body_relative_trace;
        body_relative_odom_topic_ = runtime_cfg.body_relative_odom_topic;
        enable_terminal_freeze_ = runtime_cfg.enable_terminal_freeze;
        terminal_freeze_enter_pos_ = runtime_cfg.terminal_freeze_enter_pos;
        terminal_freeze_enter_vel_ = runtime_cfg.terminal_freeze_enter_vel;
        terminal_freeze_require_vel_ = runtime_cfg.terminal_freeze_require_vel;
        terminal_freeze_exit_pos_ = runtime_cfg.terminal_freeze_exit_pos;
        command_paused_.store(runtime_cfg.start_paused);
        last_command_seq_ = runtime_cfg.command_seq;

        hover_target_.x() = runtime_cfg.hover_target_x;
        hover_target_.y() = runtime_cfg.hover_target_y;
        hover_target_.z() = runtime_cfg.hover_target_z;

        if (runtime_cfg.isConfigured()) {
            ocp_dt_ = runtime_cfg.ocp_dt;
            {
                const auto& desc = OCPRegistry::getDescriptor(ocp_type_);
                command_mode_ = desc.command_mode;
                state_dim_ = desc.state_dim;
                control_dim_ = desc.control_dim;
                custom_make_hover_state_ = desc.make_hover_state;
            }
            applyOcpDroneOdomMode(false);
            setTerminalTarget(hover_target_);
            rebuildMpcSolver();
            is_configured_ = true;
        } else {
            ocp_dt_ = 0.0;
            drone_state_is_relative_ = false;
            drone_odom_mode_ = OCPDescriptor::DroneOdomMode::Absolute;
            drone_odom_topic_.clear();
            is_configured_ = false;
        }

        sensor_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        solver_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        replay_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        if (platform_ == "crazyflie") {
            platform::crazyflie::setup(
                this, sensor_cb_group_, drone_name_, drone_odom_topic_,
                state_monitor_.crazyflieState(), state_monitor_.stateMutex(), cf_handles_);
        } else if (platform_ == "mavros") {
            platform::mavros::setup(
                this, sensor_cb_group_, hover_thrust_param_,
                state_monitor_.crazyflieState().current,  // reuse 13D ENU state slot
                state_monitor_.stateMutex(), mavros_handles_);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unknown platform: %s", platform_.c_str());
            throw std::runtime_error("Unknown platform: " + platform_);
        }

        // Keep target subscriptions alive regardless of active OCP type,
        // so runtime OCP switching to stateswitch works without restart.
        platform::target_tracker::setup(
            this, sensor_cb_group_,
            target_odom_topic_, target_accel_topic_, target_predicted_accel_topic_,
            state_monitor_.targetState(), state_monitor_.targetMutex(), target_handles_);

        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);
        debug_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/" + drone_name_ + "/planner_debug_markers", 10);

        if (is_configured_) {
            createModeTimers();
        }

        param_cb_handle_ = this->add_on_set_parameters_callback(
            std::bind(&PlannerNode::onSetParameters, this, std::placeholders::_1));

        if (is_configured_) {
            RCLCPP_INFO(this->get_logger(),
                "Ready mode=%s platform=%s solver=%s ocp=%s ocp_dt=%.3fs n_replay=%d paused=%s",
                mode_.c_str(), platform_.c_str(), solver_type_.c_str(),
                ocp_type_.c_str(), ocp_dt_, n_replay_, command_paused_.load() ? "true" : "false");
            RCLCPP_INFO(this->get_logger(),
                "Command profile target: [%.3f, %.3f, %.3f] mass=%.4f mode=%s thrust=crazysim-sysid",
                hover_target_.x(), hover_target_.y(), hover_target_.z(), mass_kg_,
                commandModeName(command_mode_));
        } else {
            RCLCPP_WARN(this->get_logger(),
                "Planner started UNCONFIGURED. Use ocp_launch.py to set ocp_type and mode.");
        }
        RCLCPP_INFO(this->get_logger(),
            "To run next OCP from another terminal: set ocp/mode/n_replay/target then increment command_seq.");
        RCLCPP_INFO(this->get_logger(),
            "Input state mode: %s (drone_odom_topic='%s')",
            droneOdomModeName(drone_odom_mode_),
            drone_odom_topic_.c_str());
    }

private:
    enum class OpenLoopPhase {
        Idle,
        Replay,
        Hold,
    };

    static const char* openLoopPhaseName(OpenLoopPhase phase) {
        switch (phase) {
            case OpenLoopPhase::Replay: return "replay";
            case OpenLoopPhase::Hold: return "hold";
            case OpenLoopPhase::Idle:
            default:
                return "idle";
        }
    }

    void stopReplayTimer() {
        if (replay_timer_) {
            replay_timer_->cancel();
            replay_timer_.reset();
        }
    }

    void stopOpenLoopHoldTimer() {
        if (open_loop_hold_timer_) {
            open_loop_hold_timer_->cancel();
            open_loop_hold_timer_.reset();
        }
    }

    void setOpenLoopPhase(OpenLoopPhase phase) {
        open_loop_phase_ = phase;
    }

    void startOpenLoopHoldTimer() {
        if (open_loop_hold_timer_) {
            return;
        }
        const int hold_ms = static_cast<int>(std::round(std::max(ocp_dt_, 0.02) * 1000.0));
        open_loop_hold_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(hold_ms),
            std::bind(&PlannerNode::openLoopHoldTick, this),
            replay_cb_group_);
    }

    static const char* commandModeName(OCPDescriptor::CommandMode mode) {
        switch (mode) {
            case OCPDescriptor::CommandMode::CmdBodyRate:
                return "cmd_bodyrate";
            case OCPDescriptor::CommandMode::CmdFullState:
            default:
                return "cmd_full_state";
        }
    }

    static const char* droneOdomModeName(OCPDescriptor::DroneOdomMode mode) {
        switch (mode) {
            case OCPDescriptor::DroneOdomMode::BodyFrameRelative:
                return "body-frame-relative";
            case OCPDescriptor::DroneOdomMode::TargetFrameRelative:
                return "target-frame-relative";
            case OCPDescriptor::DroneOdomMode::AbsoluteShiftedTarget:
                return "absolute-shifted-target";
            case OCPDescriptor::DroneOdomMode::Absolute:
            default:
                return "absolute";
        }
    }

    planner_core::Platform corePlatform() const {
        if (platform_ == "mavros") {
            return planner_core::Platform::Mavros;
        }
        return planner_core::Platform::Crazyflie;
    }

    static const char* diagnosticSeverityName(planner_core::DiagnosticSeverity severity) {
        switch (severity) {
            case planner_core::DiagnosticSeverity::Error:
                return "error";
            case planner_core::DiagnosticSeverity::Warn:
                return "warn";
            case planner_core::DiagnosticSeverity::Info:
            default:
                return "info";
        }
    }

    static std::array<float, 4> hoverArray(const Eigen::Vector4d& hover) {
        return {float(hover(0)), float(hover(1)), float(hover(2)), float(hover(3))};
    }

    bool shouldDisarmOnLandingFinish() const {
        if (platform_ != "crazyflie" || ocp_type_.empty()) {
            return false;
        }
        return OCPRegistry::getDescriptor(ocp_type_).disarm_on_landing_finish;
    }

    bool currentStateInsideTerminalPositionThreshold(
        const Eigen::VectorXd& current_state,
        const TargetSnapshot& target_snapshot) const
    {
        const auto& desc = OCPRegistry::getDescriptor(ocp_type_);
        Eigen::VectorXd terminal_state = current_state;
        if (desc.transform_state) {
            if (!target_snapshot.valid) {
                return false;
            }
            terminal_state = desc.transform_state(current_state, target_snapshot);
        }
        if (terminal_state.size() < 3) {
            return false;
        }
        if (desc.transform_state) {
            return terminal_state.segment(0, 3).norm() < terminal_freeze_enter_pos_;
        }
        return (terminal_state.segment(0, 3) - terminal_position_abs_).norm() <
            terminal_freeze_enter_pos_;
    }

    bool rejectedSolveCompletesLanding(
        const planner_core::PlannerCoreStepResult& result,
        const Eigen::VectorXd& current_state,
        const TargetSnapshot& target_snapshot) const
    {
        return shouldDisarmOnLandingFinish() &&
            result.solve_attempted &&
            result.rejection_reason == "constraint_error" &&
            currentStateInsideTerminalPositionThreshold(current_state, target_snapshot);
    }

    bool dispatchLandingShutdownOnce(const std::string& reason) {
        if (!shouldDisarmOnLandingFinish()) {
            return false;
        }
        if (landing_shutdown_sent_.load()) {
            return true;
        }
        if (!platform::crazyflie::requestLandingShutdownAsync(this, cf_handles_, reason)) {
            return false;
        }
        landing_shutdown_sent_.store(true);
        command_paused_.store(true);
        ocp_active_.store(false);
        maintain_hover_hold_ = false;
        terminal_freeze_.store(true);
        return true;
    }

    geometry_msgs::msg::Point pointMsg(const Eigen::Vector3d& p) const {
        geometry_msgs::msg::Point out;
        out.x = p.x();
        out.y = p.y();
        out.z = p.z();
        return out;
    }

    Eigen::Vector3d commandWorldPosition(const Eigen::VectorXd& state,
                                         const TargetSnapshot& target) const {
        if (state.size() < 3) {
            return Eigen::Vector3d::Zero();
        }
        const auto& desc = OCPRegistry::getDescriptor(ocp_type_);
        if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative) {
            const Eigen::Matrix3d R_WN = Quad6DOFVarTime<double>::calcC(target.orientation);
            return target.position + R_WN * state.segment(0, 3);
        }
        if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative &&
            state.size() >= 10) {
            const Eigen::Vector4d q_NB = state.segment(6, 4);
            const Eigen::Matrix3d R_WN = Quad6DOFVarTime<double>::calcC(target.orientation);
            const Eigen::Matrix3d R_WB = R_WN * Quad6DOFVarTime<double>::calcC(q_NB);
            return target.position - R_WB * state.segment(0, 3);
        }
        return state.segment(0, 3);
    }

    visualization_msgs::msg::Marker baseMarker(int id,
                                               const std::string& ns,
                                               int type) const {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = this->now();
        marker.ns = ns;
        marker.id = id;
        marker.type = type;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.lifetime = rclcpp::Duration::from_seconds(0.5);
        return marker;
    }

    void applyOcpDroneOdomMode(bool rebuild_subscriptions) {
        if (ocp_type_.empty()) {
            return;
        }

        const auto& desc = OCPRegistry::getDescriptor(ocp_type_);
        const bool needs_relative =
            (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative) ||
            (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative);

        std::string desired_odom_topic;
        if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative)
            desired_odom_topic = body_relative_odom_topic_;
        else if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative)
            desired_odom_topic = target_frame_odom_topic_;
        else
            desired_odom_topic = "";

        const bool mode_changed = (needs_relative != drone_state_is_relative_);
        const bool topic_changed = (desired_odom_topic != drone_odom_topic_);

        drone_odom_mode_ = desc.drone_odom_mode;
        drone_state_is_relative_ = needs_relative;
        drone_odom_topic_ = desired_odom_topic;

        if (!rebuild_subscriptions || platform_ != "crazyflie" || (!mode_changed && !topic_changed)) {
            return;
        }

        cf_handles_.reset();
        state_monitor_.crazyflieState().reset();
        platform::crazyflie::setup(
            this,
            sensor_cb_group_,
            drone_name_,
            drone_odom_topic_,
            state_monitor_.crazyflieState(),
            state_monitor_.stateMutex(),
            cf_handles_);

        RCLCPP_INFO(this->get_logger(),
            "[OCP switch] drone odom mode=%s topic='%s'",
            droneOdomModeName(desc.drone_odom_mode),
            drone_odom_topic_.c_str());
    }

    void transitionOpenLoopToHold(const std::string& reason) {
        stopReplayTimer();
        command_paused_.store(true);
        maintain_hover_hold_ = true;
        setOpenLoopPhase(OpenLoopPhase::Hold);
        startOpenLoopHoldTimer();
        RCLCPP_INFO(this->get_logger(),
            "[OpenLoop] Replay complete (%s). Transitioning to hold phase.",
            reason.c_str());
        openLoopHoldTick();
    }

    void abortOpenLoopReplay(const std::string& reason) {
        stopReplayTimer();
        setOpenLoopPhase(OpenLoopPhase::Idle);
        command_paused_.store(true);
        maintain_hover_hold_ = false;

        resetForNewCommand();
        ocp_active_.store(false);
        RCLCPP_WARN(this->get_logger(),
            "[OpenLoop] Replay aborted: %s. Replay stopped with no hover fallback command.",
            reason.c_str());
    }

    bool hasState() const {
        if (platform_ == "mavros") {
            return mavros_handles_.odom_received;
        }
        return state_monitor_.hasState(platform_);
    }

    bool hasFreshTargetState() const {
        if (ocp_type_.empty()) {
            return true;
        }
        const auto desc = OCPRegistry::getDescriptor(ocp_type_);
        if (!desc.validate_target) {
            return true;
        }
        const double now_sec = this->now().seconds();
        TargetSnapshot snapshot = state_monitor_.getTargetSnapshot(true, this->now());
        return desc.validate_target(snapshot, now_sec, 0.2);
    }

    TargetSnapshot getTargetSnapshot() const {
        if (ocp_type_.empty()) return TargetSnapshot{true};
        bool needs_target = OCPRegistry::getDescriptor(ocp_type_).validate_target != nullptr;
        return state_monitor_.getTargetSnapshot(needs_target, this->now());
    }

    bool isTargetValidForDescriptor(const OCPDescriptor& desc,
                                    const TargetSnapshot& snapshot) const {
        if (desc.validate_target) {
            return desc.validate_target(snapshot, this->now().seconds(), 0.2);
        }
        return snapshot.valid;
    }

    bool tryGetTrackingCircleTargetExtra(
        const std::any& extra,
        TrackingCircleTargetOCP::TrackingCircleTargetExtra& out) const {
        if (!extra.has_value()) {
            return false;
        }
        try {
            out = std::any_cast<TrackingCircleTargetOCP::TrackingCircleTargetExtra>(extra);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }

    double stateNodeTime(const Eigen::VectorXd& s, int index) const {
        if (s.size() > state_dim_) {
            return s(state_dim_);
        }
        return index * ocp_dt_;
    }

    double replayAdvanceTime(const std::vector<Eigen::VectorXd>& traj) const {
        if (traj.size() < 2) {
            return std::max(ocp_dt_, 1e-3);
        }
        const int idx = std::min(n_replay_, static_cast<int>(traj.size()) - 1);
        const double t_adv = stateNodeTime(traj[idx], idx);
        return std::max(t_adv, std::max(ocp_dt_, 1e-3));
    }

    void integrateTargetSegment(Eigen::Vector3d& pos,
                                Eigen::Vector3d& vel,
                                double t_abs_start,
                                double t_abs_end,
                                const target_models::TargetAccelBuffer& buffer) const {
        if (t_abs_end <= t_abs_start) {
            return;
        }
        const double nominal_step = (buffer.dt > 1e-6) ? buffer.dt : 0.05;
        const double integration_step = std::clamp(nominal_step, 0.005, 0.05);

        double t = t_abs_start;
        while (t < t_abs_end - 1e-9) {
            const double h = std::min(integration_step, t_abs_end - t);
            const Eigen::Vector3d a = buffer.getAccel(t);
            pos += vel * h + 0.5 * a * h * h;
            vel += a * h;
            t += h;
        }
    }

    // tracking_circle_target OCP uses only future target acceleration.
    // Current target pose/velocity is still required here so planner-side
    // interface plumbing can reconstruct absolute target motion for command
    // publishing and logs. This is not part of the OCP model itself.
    std::vector<Eigen::VectorXd> reconstructTrackingCircleTargetAbsoluteTrajectory(
        const std::vector<Eigen::VectorXd>& rel_traj,
        const Eigen::Vector3d& target_pos0,
        const Eigen::Vector3d& target_vel0,
        const target_models::TargetAccelBuffer& buffer,
        double t0_abs,
        std::vector<Eigen::Vector3d>* target_pos_nodes = nullptr,
        std::vector<Eigen::Vector3d>* target_vel_nodes = nullptr) const {
        std::vector<Eigen::VectorXd> abs_traj;
        abs_traj.reserve(rel_traj.size());
        if (rel_traj.empty()) {
            return abs_traj;
        }

        if (target_pos_nodes) {
            target_pos_nodes->clear();
            target_pos_nodes->reserve(rel_traj.size());
        }
        if (target_vel_nodes) {
            target_vel_nodes->clear();
            target_vel_nodes->reserve(rel_traj.size());
        }

        Eigen::Vector3d tgt_pos = target_pos0;
        Eigen::Vector3d tgt_vel = target_vel0;
        double prev_t_node = 0.0;

        for (int i = 0; i < static_cast<int>(rel_traj.size()); ++i) {
            const Eigen::VectorXd& x_rel = rel_traj[i];
            if (x_rel.size() < state_dim_) {
                if (target_pos_nodes) {
                    target_pos_nodes->push_back(tgt_pos);
                }
                if (target_vel_nodes) {
                    target_vel_nodes->push_back(tgt_vel);
                }
                abs_traj.push_back(x_rel);
                continue;
            }

            const double raw_t_node = stateNodeTime(x_rel, i);
            const double t_node = std::max(prev_t_node, raw_t_node);
            integrateTargetSegment(tgt_pos, tgt_vel,
                                   t0_abs + prev_t_node,
                                   t0_abs + t_node,
                                   buffer);
            prev_t_node = t_node;

            if (target_pos_nodes) {
                target_pos_nodes->push_back(tgt_pos);
            }
            if (target_vel_nodes) {
                target_vel_nodes->push_back(tgt_vel);
            }

            Eigen::VectorXd x_abs = x_rel;
            x_abs.segment(0, 3) = x_rel.segment(0, 3) + tgt_pos;
            x_abs.segment(3, 3) = x_rel.segment(3, 3) + tgt_vel;
            abs_traj.push_back(x_abs);
        }

        return abs_traj;
    }

    Eigen::VectorXd getCurrentState() const {
        if (platform_ == "mavros") {
            std::lock_guard<std::mutex> lk(state_monitor_.stateMutex());
            return mavros_handles_.current;
        }
        return state_monitor_.getCurrentState(platform_);
    }

    // Convert only sensor states that are explicitly published in shifted coordinates.
    // Stateswitch receives world-frame drone state and shifts it only inside the OCP
    // transform_state callback, so adding the live target here would create a bogus
    // hover/full-state command after terminal freeze.
    Eigen::VectorXd convertStateToAbsoluteFrame(const Eigen::VectorXd& state) const {
        if (drone_state_is_relative_ || state.size() < state_dim_ ||
            drone_odom_mode_ != OCPDescriptor::DroneOdomMode::AbsoluteShiftedTarget) {
            return state;
        }

        Eigen::VectorXd x_abs = state;
        const auto [tgt_pos, tgt_vel] = state_monitor_.getTargetPositionVelocity();
        x_abs.segment(0, 3) += tgt_pos;
        x_abs.segment(3, 3) += tgt_vel;
        return x_abs;
    }

    const char* currentFrameLabel() const {
        return planner_core::frameLabel(drone_odom_mode_);
    }

    std::vector<Eigen::VectorXd> makeTrajectoryForPublishing(const SolverResult& result) const {
        // BodyRelative mode: reconstruct the drone world trajectory if we have
        // the target world trajectory for each node. Otherwise fall back to the
        // raw body-relative state for diagnostics.
        if (drone_state_is_relative_) {
            if (!result.is_relative_plan ||
                result.target_world_pos_trajectory.size() != result.state_trajectory.size() ||
                result.target_world_vel_trajectory.size() != result.state_trajectory.size()) {
                return result.state_trajectory;
            }

            std::vector<Eigen::VectorXd> traj_abs = result.state_trajectory;
            for (int i = 0; i < static_cast<int>(traj_abs.size()); ++i) {
                auto& s = traj_abs[i];
                if (s.size() < state_dim_) {
                    continue;
                }

                const Eigen::Vector3d tgt_p = result.target_world_pos_trajectory[i];
                const Eigen::Vector3d tgt_v = result.target_world_vel_trajectory[i];
                const Eigen::Vector4d q_NB = s.segment(6, 4);
                const double t_node = stateNodeTime(s, i);
                const Eigen::AngleAxisd yaw_step(
                    result.target_snapshot_omega.z() * t_node, Eigen::Vector3d::UnitZ());
                const Eigen::Matrix3d R_WN =
                    Quad6DOFVarTime<double>::calcC(result.target_snapshot_quat) *
                    yaw_step.toRotationMatrix();
                const Eigen::Matrix3d R_WB = R_WN * Quad6DOFVarTime<double>::calcC(q_NB);
                const Eigen::Vector3d p_T_B = s.segment(0, 3);
                const Eigen::Vector3d v_rel_B = s.segment(3, 3);

                s.segment(0, 3) = tgt_p - R_WB * p_T_B;
                s.segment(3, 3) = tgt_v + R_WB * v_rel_B;
            }
            return traj_abs;
        }

        if (!result.is_relative_plan) {
            return result.state_trajectory;
        }

        if (result.target_world_pos_trajectory.size() == result.state_trajectory.size() &&
            result.target_world_vel_trajectory.size() == result.state_trajectory.size()) {
            std::vector<Eigen::VectorXd> traj_abs = result.state_trajectory;
            for (int i = 0; i < static_cast<int>(traj_abs.size()); ++i) {
                auto& s = traj_abs[i];
                if (s.size() < state_dim_) {
                    continue;
                }
                if (drone_odom_mode_ == OCPDescriptor::DroneOdomMode::TargetFrameRelative) {
                    const double t_node = stateNodeTime(s, i);
                    const Eigen::AngleAxisd yaw_step(
                        result.target_snapshot_omega.z() * t_node, Eigen::Vector3d::UnitZ());
                    const Eigen::Matrix3d R_WN =
                        Quad6DOFVarTime<double>::calcC(result.target_snapshot_quat) *
                        yaw_step.toRotationMatrix();
                    const Eigen::Vector3d p_N = s.segment(0, 3);
                    const Eigen::Vector3d v_N = s.segment(3, 3);
                    s.segment(0, 3) = result.target_world_pos_trajectory[i] + R_WN * p_N;
                    s.segment(3, 3) = result.target_world_vel_trajectory[i] +
                                      R_WN * (v_N + result.target_snapshot_omega.cross(p_N));
                } else {
                    s.segment(0, 3) += result.target_world_pos_trajectory[i];
                    s.segment(3, 3) += result.target_world_vel_trajectory[i];
                }
            }
            return traj_abs;
        }

        if (ocp_type_ == "tracking_circle_target") {
            TrackingCircleTargetOCP::TrackingCircleTargetExtra ex;
            if (tryGetTrackingCircleTargetExtra(result.extra, ex)) {
                return reconstructTrackingCircleTargetAbsoluteTrajectory(
                    result.state_trajectory,
                    result.target_snapshot_pos,
                    result.target_snapshot_vel,
                    ex.buf,
                    ex.t_abs);
            }
        }

        std::vector<Eigen::VectorXd> traj_abs = result.state_trajectory;
        for (int i = 0; i < static_cast<int>(traj_abs.size()); ++i) {
            auto& s = traj_abs[i];
            if (s.size() < state_dim_) {
                continue;
            }

            const double t_node = stateNodeTime(s, i);

            const Eigen::Vector3d tgt_p = result.target_snapshot_pos +
                                          result.target_snapshot_vel * t_node +
                                          0.5 * result.target_snapshot_acc * t_node * t_node;
            const Eigen::Vector3d tgt_v = result.target_snapshot_vel +
                                          result.target_snapshot_acc * t_node;

            s.segment(0, 3) += tgt_p;
            s.segment(3, 3) += tgt_v;
        }

        return traj_abs;
    }

    void setTerminalTarget(const Eigen::Vector3d& xyz) {
        hover_target_ = xyz;
        terminal_position_abs_ = xyz;

        Eigen::VectorXd terminal = Eigen::VectorXd::Zero(state_dim_);
        if (state_dim_ >= 7) {
            terminal(0) = xyz.x();
            terminal(1) = xyz.y();
            terminal(2) = xyz.z();
            terminal(6) = 1.0;
        }
        terminal_state_ = terminal;

        if (alipddp_mpc_) {
            alipddp_mpc_->setTerminalState(terminal_state_);
        }
        if (planner_core_configured_) {
            planner_core_.setTerminalTarget(xyz);
        }
    }

    void rebuildMpcSolver() {
        if (solver_type_ != "alipddp") {
            return;
        }
        QuadrotorMPC::Config cfg;
        cfg.ocp_type = ocp_type_;
        cfg.terminal_state = terminal_state_;
        cfg.n_shift = n_replay_;
        alipddp_mpc_ = std::make_unique<QuadrotorMPC>(cfg);

        planner_core::PlannerCoreConfig core_cfg;
        core_cfg.ocp_type = ocp_type_;
        core_cfg.platform = corePlatform();
        core_cfg.solver_type = solver_type_;
        core_cfg.ocp_dt = ocp_dt_;
        core_cfg.n_replay = n_replay_;
        core_cfg.max_constraint_error = max_constraint_error_;
        core_cfg.skip_trajectory_validation = skip_trajectory_validation_;
        core_cfg.enable_terminal_freeze = enable_terminal_freeze_;
        core_cfg.terminal_freeze_enter_pos = terminal_freeze_enter_pos_;
        core_cfg.terminal_freeze_enter_vel = terminal_freeze_enter_vel_;
        core_cfg.terminal_freeze_require_vel = terminal_freeze_require_vel_;
        core_cfg.terminal_freeze_exit_pos = terminal_freeze_exit_pos_;
        core_cfg.terminal_position_abs = terminal_position_abs_;
        core_cfg.terminal_state = terminal_state_;
        core_cfg.body_relative_odom_topic = body_relative_odom_topic_;
        core_cfg.target_frame_odom_topic = target_frame_odom_topic_;
        core_cfg.max_first_solve_age_sec = max_first_solve_age_sec_;
        core_cfg.max_relative_position_norm = max_relative_position_norm_;
        core_cfg.max_relative_vertical_abs = max_relative_vertical_abs_;
        core_cfg.max_relative_velocity_norm = max_relative_velocity_norm_;
        planner_core_.configure(core_cfg);
        planner_core_configured_ = true;
    }

    void createModeTimers() {
        if (mode_ == "mpc") {
            if (!solver_timer_) {
                solver_timer_ = this->create_wall_timer(
                    1ms,
                    std::bind(&PlannerNode::solverLoop, this),
                    solver_cb_group_);
            }
            if (!mpc_replay_timer_ && ocp_dt_ > 0) {
                const int replay_ms = static_cast<int>(std::round(ocp_dt_ * 1000.0));
                mpc_replay_timer_ = this->create_wall_timer(
                    std::chrono::milliseconds(replay_ms),
                    std::bind(&PlannerNode::mpcReplayTick, this),
                    replay_cb_group_);
            }
        } else if (mode_ == "open_loop") {
            if (!startup_timer_) {
                startup_timer_ = this->create_wall_timer(
                    50ms,
                    std::bind(&PlannerNode::openLoopStartupCheck, this),
                    solver_cb_group_);
            }
        }
    }

    void resetForNewCommand() {
        stopReplayTimer();
        stopOpenLoopHoldTimer();
        setOpenLoopPhase(OpenLoopPhase::Idle);
        planner_core_.reset();
        is_primed_.store(false);
        terminal_freeze_.store(false);
        landing_shutdown_sent_.store(false);
        stale_warning_count_ = 0;
        replay_ticks_since_solve_.store(0);
        last_accepted_solve_timestamp_ = Clock::time_point{};
        last_replan_delay_sec_ = 0.0;
    }

    // Publishes a world-frame state directly without applying reconstruct_world_state.
    // Use for hover-hold ticks where paused_hover_state_ is already in world frame.
    void publishCommandAbsolute(const Eigen::VectorXd& s_world, const Eigen::VectorXd& u) {
        if (landing_shutdown_sent_.load()) {
            return;
        }
        if (platform_ == "crazyflie") {
            platform::crazyflie::publishCommand(this, cf_handles_, s_world, u, mass_kg_);
        } else if (platform_ == "mavros") {
            platform::mavros::publishFullStateCommand(mavros_handles_, s_world);
        }
    }

    void publishPausedHoverHoldTick() {
        if (landing_shutdown_sent_.load()) {
            return;
        }
        if (command_mode_ == OCPDescriptor::CommandMode::CmdBodyRate) {
            if (platform_ == "crazyflie") {
                if (drone_state_is_relative_) {
                    return; // Let goto hold the drone, we cannot give an absolute z before the first solve
                }
                // Hold altitude using cmd_hover with zero velocity
                const Eigen::VectorXd x_abs = convertStateToAbsoluteFrame(getCurrentState());
                const float z_hold = (x_abs.size() >= 3) ? float(x_abs(2)) : 0.3f;
                platform::crazyflie::publishHoverCommandDirect(cf_handles_,
                    {0.0f, 0.0f, z_hold, 0.0f});
            } else if (platform_ == "mavros") {
                // MAVROS keepalive handled by heartbeat_timer in mavros.hpp
            }
            return;
        }
        if (drone_state_is_relative_) {
            return;
        }

        const Eigen::VectorXd x_now_abs = convertStateToAbsoluteFrame(getCurrentState());
        hover_controller::doHoverHoldTick(
            is_configured_,
            mass_kg_,
            state_dim_,
            custom_make_hover_state_,
            maintain_hover_hold_,
            hasState(),
            x_now_abs,
            paused_hover_state_,
            [this](const auto& x, const auto& u) { publishCommandAbsolute(x, u); }
        );
        if (ocp_active_.load()) {
            planner_logging::SolverEventLogRow row;
            row.solve_num = -1;
            row.event = "stale";
            row.reason = "paused_hover_hold_tick";
            row.coord_mode = "absolute";
            logger_.logSolverEvent(row);
        }
    }

    void holdHoverAndPause(const std::string& reason) {
        if (drone_state_is_relative_) {
            // BodyRelative mode: hover_controller needs world-frame state which the
            // planner does not have.  Just pause and let publishPausedHoverHoldTick
            // send safe zero-velocity cmd_hover at the configured altitude.
            RCLCPP_WARN(this->get_logger(),
                "holdHoverAndPause(%s) in body-relative mode — pausing without world-frame hold.",
                reason.c_str());
            command_paused_.store(true);
            maintain_hover_hold_ = false;
            return;
        }

        hover_controller::enterHoverHold(
            reason,
            is_configured_,
            mass_kg_,
            state_dim_,
            custom_make_hover_state_,
            hasState(),
            convertStateToAbsoluteFrame(getCurrentState()),
            command_paused_,
            maintain_hover_hold_,
            paused_hover_state_,
            [this](const auto& x, const auto& u) { publishCommandAbsolute(x, u); },
            [this](const char* r) {
                RCLCPP_INFO(this->get_logger(),
                    "Command finished (%s). Holding hover and waiting for next command_seq.", r);
            },
            [this]() {
                resetForNewCommand();
                ocp_active_.store(false);
            }
        );
    }

    void logCoreDiagnostics(const std::vector<planner_core::PlannerDiagnostic>& diagnostics) {
        for (const auto& d : diagnostics) {
            if (d.code == "handoff") {
                const auto value = [&](const char* key) {
                    const auto it = d.values.find(key);
                    return (it == d.values.end()) ? 0.0 : it->second;
                };
                RCLCPP_INFO(this->get_logger(),
                    "[handoff] solve %.0f -> %.0f age=%.3fs latency=%.3fs state_jump=%.4f control_jump=%.4f hover_z_jump=%.4f",
                    value("previous_solve_num"), value("new_solve_num"),
                    value("active_plan_age_sec"), value("solve_latency_sec"),
                    value("state_jump_norm"), value("control_jump_norm"),
                    value("hover_z_jump"));
                continue;
            }
            if (d.code == "solve_rejected") {
                const auto value = [&](const char* key) {
                    const auto it = d.values.find(key);
                    return (it == d.values.end()) ? 0.0 : it->second;
                };
                RCLCPP_WARN(this->get_logger(),
                    "[PlannerCore][solve_rejected] %s solve=%.1fms constraint=%.3g "
                    "max_rel_p=%.3f(node %.0f)/limit %.3f max_rel_z=%.3f(node %.0f)/limit %.3f "
                    "max_rel_v=%.3f(node %.0f)/limit %.3f",
                    d.message.c_str(),
                    value("solve_time_ms"),
                    value("constraint_error"),
                    value("max_relative_position_norm"),
                    value("max_relative_position_node"),
                    value("relative_position_limit"),
                    value("max_relative_vertical_abs"),
                    value("max_relative_vertical_node"),
                    value("relative_vertical_limit"),
                    value("max_relative_velocity_norm"),
                    value("max_relative_velocity_node"),
                    value("relative_velocity_limit"));
                continue;
            }
            if (d.severity == planner_core::DiagnosticSeverity::Error) {
                RCLCPP_ERROR(this->get_logger(), "[PlannerCore][%s] %s",
                    d.code.c_str(), d.message.c_str());
            } else if (d.severity == planner_core::DiagnosticSeverity::Warn) {
                RCLCPP_WARN(this->get_logger(), "[PlannerCore][%s] %s",
                    d.code.c_str(), d.message.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "[PlannerCore][%s] %s",
                    d.code.c_str(), d.message.c_str());
            }
            (void)diagnosticSeverityName(d.severity);
        }
    }

    bool ensureLoggingInitialized() {
        if (!logging_enabled_) {
            return false;
        }
        if (logging_initialized_) {
            return true;
        }
        if (!is_configured_ || ocp_type_.empty() || mode_.empty()) {
            return false;
        }
        auto dec = OCPRegistry::getDescriptor(ocp_type_);
        logging_initialized_ = logger_.initialize(
            drone_name_, ocp_type_, mode_, solver_type_, this->get_logger(), mass_kg_,
            state_dim_, dec.state_names, dec.control_names, dec.log_state_headers,
            command_mode_, dec.extract_actual_state_row);
        return logging_initialized_;
    }

    void publishDebugMarkers(const planner_core::PlannerCommand& command,
                             const std::vector<planner_core::PlannerDiagnostic>& diagnostics) {
        if (!debug_marker_pub_) {
            return;
        }

        target_history_.push_back(last_target_snapshot_.position);
        while (target_history_.size() > 250) {
            target_history_.pop_front();
        }

        visualization_msgs::msg::MarkerArray array;

        auto target = baseMarker(0, "target", visualization_msgs::msg::Marker::SPHERE);
        target.pose.position = pointMsg(last_target_snapshot_.position);
        target.scale.x = 0.12;
        target.scale.y = 0.12;
        target.scale.z = 0.12;
        target.color.r = 1.0f;
        target.color.g = 0.82f;
        target.color.b = 0.10f;
        target.color.a = 0.95f;
        array.markers.push_back(target);

        auto target_path = baseMarker(1, "target_path", visualization_msgs::msg::Marker::LINE_STRIP);
        target_path.scale.x = 0.025;
        target_path.color.r = 1.0f;
        target_path.color.g = 0.72f;
        target_path.color.b = 0.10f;
        target_path.color.a = 0.75f;
        for (const auto& p : target_history_) {
            target_path.points.push_back(pointMsg(p));
        }
        array.markers.push_back(target_path);

        if (command.state.size() >= 3) {
            const Eigen::Vector3d cmd_pos = commandWorldPosition(command.state, last_target_snapshot_);
            auto cmd = baseMarker(2, "active_command", visualization_msgs::msg::Marker::SPHERE);
            cmd.pose.position = pointMsg(cmd_pos);
            cmd.scale.x = 0.10;
            cmd.scale.y = 0.10;
            cmd.scale.z = 0.10;
            cmd.color.r = 0.10f;
            cmd.color.g = 0.50f;
            cmd.color.b = 1.0f;
            cmd.color.a = 0.95f;
            array.markers.push_back(cmd);
        }

        for (const auto& d : diagnostics) {
            if (d.code != "handoff" || d.state_a.size() < 3 || d.state_b.size() < 3) {
                continue;
            }
            const Eigen::Vector3d old_pos = commandWorldPosition(d.state_a, last_target_snapshot_);
            const Eigen::Vector3d new_pos = commandWorldPosition(d.state_b, last_target_snapshot_);
            auto line = baseMarker(3, "handoff_jump", visualization_msgs::msg::Marker::LINE_LIST);
            line.scale.x = 0.04;
            line.color.r = 1.0f;
            line.color.g = 0.15f;
            line.color.b = 0.05f;
            line.color.a = 0.95f;
            line.points.push_back(pointMsg(old_pos));
            line.points.push_back(pointMsg(new_pos));
            line.lifetime = rclcpp::Duration::from_seconds(3.0);
            array.markers.push_back(line);

            const Eigen::Vector3d mid = 0.5 * (old_pos + new_pos);
            auto text = baseMarker(4, "handoff_text", visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
            text.pose.position = pointMsg(mid + Eigen::Vector3d(0.0, 0.0, 0.15));
            text.scale.z = 0.12;
            text.color.r = 1.0f;
            text.color.g = 0.20f;
            text.color.b = 0.05f;
            text.color.a = 1.0f;
            const auto value = [&](const char* key) {
                const auto it = d.values.find(key);
                return (it == d.values.end()) ? 0.0 : it->second;
            };
            std::ostringstream label;
            label << "handoff dx=" << std::fixed << std::setprecision(3)
                  << value("state_jump_norm")
                  << " du=" << value("control_jump_norm");
            text.text = label.str();
            text.lifetime = rclcpp::Duration::from_seconds(3.0);
            array.markers.push_back(text);
        }

        debug_marker_pub_->publish(array);
    }

    planner_core::PlannerCoreInput makeCoreInput(const Eigen::VectorXd& state,
                                                 const TargetSnapshot& target_snapshot) {
        planner_core::PlannerCoreInput input;
        input.current_state = state;
        input.target_snapshot = target_snapshot;
        input.active_odom_mode = drone_odom_mode_;
        input.active_odom_topic = drone_odom_topic_;
        input.now = Clock::now();
        input.ros_time_sec = this->now().seconds();
        input.target_accel_buffer = state_monitor_.getTargetAccelBuffer();
        input.has_target_accel_buffer = state_monitor_.hasTargetTrajectory(this->now());
        return input;
    }

    planner_logging::SolveLogMeta toRosLogMeta(const planner_core::PlannerSolveLogData& data,
                                               const SolverResult& solve_result) {
        planner_logging::SolveLogMeta meta;
        meta.solve_num = data.solve_num;
        meta.solve_time_ms = data.solve_time_ms;
        meta.solve_iters = data.solve_iters;
        meta.is_relative_plan = data.is_relative_plan;
        meta.ocp_dt = data.ocp_dt;
        meta.coord_mode = data.coord_mode;
        meta.target_snapshot_pos = data.target_snapshot_pos;
        meta.target_snapshot_vel = data.target_snapshot_vel;
        meta.target_snapshot_acc = data.target_snapshot_acc;
        meta.target_snapshot_quat = data.target_snapshot_quat;
        meta.target_snapshot_omega = data.target_snapshot_omega;
        meta.target_snapshot_beta = data.target_snapshot_beta;
        meta.target_world_pos_trajectory = data.target_world_pos_trajectory;
        meta.target_world_vel_trajectory = data.target_world_vel_trajectory;
        auto desc = OCPRegistry::getDescriptor(ocp_type_);
        if (desc.prepare_log_meta) {
            desc.prepare_log_meta(meta, solve_result.extra, runtime_cfg_);
        }
        return meta;
    }

    static double diagnosticValue(const planner_core::PlannerDiagnostic& d,
                                  const std::string& key,
                                  double fallback = std::numeric_limits<double>::quiet_NaN()) {
        const auto it = d.values.find(key);
        return (it == d.values.end()) ? fallback : it->second;
    }

    static const planner_core::PlannerDiagnostic* findDiagnostic(
        const std::vector<planner_core::PlannerDiagnostic>& diagnostics,
        const std::string& code) {
        for (const auto& d : diagnostics) {
            if (d.code == code) {
                return &d;
            }
        }
        return nullptr;
    }

    void logSolverEvents(const planner_core::PlannerCoreStepResult& result) {
        if (!logging_enabled_ || !logging_initialized_) {
            return;
        }

        auto coord_mode = [&]() -> std::string {
            if (!result.solve_log.coord_mode.empty()) {
                return result.solve_log.coord_mode;
            }
            if (result.solve_result.is_relative_plan) {
                return "absolute_shifted";
            }
            return currentFrameLabel();
        };

        if (result.solve_accepted) {
            planner_logging::SolverEventLogRow row;
            row.solve_num = result.solve_num;
            row.event = "accepted";
            row.coord_mode = coord_mode();
            row.x0_source = result.x0_source;
            row.solve_time_ms = result.solve_result.solve_time_ms;
            row.solve_iters = static_cast<double>(result.solve_result.solve_iters);
            row.constraint_error = result.solve_result.constraint_error;
            row.active_elapsed_now_sec = result.active_elapsed_now_sec;
            row.activation_elapsed_sec = result.activation_elapsed_sec;
            row.activation_wall_time_sec = result.activation_wall_time_sec;
            row.solve_lead_sec = result.solve_lead_sec;
            row.solve_finish_late_by_sec = result.solve_finish_late_by_sec;
            row.handoff_pos_err = result.handoff_pos_err;
            row.handoff_vel_err = result.handoff_vel_err;
            row.replan_delay_sec = result.replan_delay_sec;
            logger_.logSolverEvent(row);
        } else if (result.solve_attempted) {
            planner_logging::SolverEventLogRow row;
            row.solve_num = result.solve_num;
            row.event = "rejected";
            row.reason = result.rejection_reason;
            row.coord_mode = coord_mode();
            row.x0_source = result.x0_source;
            row.solve_time_ms = result.solve_result.solve_time_ms;
            row.solve_iters = static_cast<double>(result.solve_result.solve_iters);
            row.constraint_error = result.solve_result.constraint_error;
            row.active_elapsed_now_sec = result.active_elapsed_now_sec;
            row.activation_elapsed_sec = result.activation_elapsed_sec;
            row.activation_wall_time_sec = result.activation_wall_time_sec;
            row.solve_lead_sec = result.solve_lead_sec;
            row.solve_finish_late_by_sec = result.solve_finish_late_by_sec;
            row.handoff_pos_err = result.handoff_pos_err;
            row.handoff_vel_err = result.handoff_vel_err;
            if (const auto* d = findDiagnostic(result.diagnostics, "solve_rejected")) {
                row.extra_values = d->values;
            }
            logger_.logSolverEvent(row);
        }

        for (const auto& d : result.diagnostics) {
            if (d.code != "handoff_recovery_live") {
                continue;
            }
            planner_logging::SolverEventLogRow row;
            row.solve_num = result.solve_num;
            row.event = "recovery";
            row.reason = d.message;
            row.coord_mode = coord_mode();
            row.x0_source = result.x0_source;
            row.handoff_pos_err = diagnosticValue(d, "handoff_pos_err_now");
            row.handoff_vel_err = diagnosticValue(d, "handoff_vel_err_now");
            row.activation_elapsed_sec = diagnosticValue(d, "activation_elapsed");
            row.extra_values = d.values;
            logger_.logSolverEvent(row);
        }
    }

    void logReplaySolverEvents(const planner_core::PlannerCoreStepResult& replay) {
        if (!logging_enabled_ || !logging_initialized_) {
            return;
        }
        if (replay.stale) {
            planner_logging::SolverEventLogRow row;
            row.solve_num = replay.solve_num;
            row.event = "stale";
            row.reason = "trajectory_stale";
            row.coord_mode = currentFrameLabel();
            if (const auto* d = findDiagnostic(replay.diagnostics, "trajectory_stale")) {
                row.extra_values = d->values;
            }
            logger_.logSolverEvent(row);
        }
        for (const auto& d : replay.diagnostics) {
            if (d.code != "handoff") {
                continue;
            }
            planner_logging::SolverEventLogRow row;
            row.solve_num = static_cast<int>(diagnosticValue(d, "new_solve_num", replay.solve_num));
            row.event = "handoff";
            row.reason = d.message;
            row.coord_mode = currentFrameLabel();
            row.state_jump_norm = diagnosticValue(d, "state_jump_norm");
            row.control_jump_norm = diagnosticValue(d, "control_jump_norm");
            row.hover_z_jump = diagnosticValue(d, "hover_z_jump");
            row.solve_time_ms = diagnosticValue(d, "solve_latency_sec") * 1000.0;
            row.extra_values = d.values;
            logger_.logSolverEvent(row);
        }
    }

    rcl_interfaces::msg::SetParametersResult onSetParameters(
    const std::vector<rclcpp::Parameter>& params) {
        std::lock_guard<std::mutex> lk(command_mutex_);

        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "ok";

        std::string new_ocp_type = ocp_type_;
        std::string new_mode = mode_;
        int new_n_replay = n_replay_;
        Eigen::Vector3d new_target = hover_target_;
        int new_command_seq = last_command_seq_;

        bool profile_changed = false;
        bool ocp_type_changed = false;

        for (const auto& p : params) {
            if (p.get_name() == "ocp_type") {
                new_ocp_type = p.as_string();
                try {
                    (void)OCPRegistry::getDT(new_ocp_type);
                } catch (const std::exception& e) {
                    result.successful = false;
                    result.reason = e.what();
                    return result;
                }
                profile_changed = true;
                ocp_type_changed = true;
            } else if (p.get_name() == "mode") {
                new_mode = p.as_string();
                if (new_mode != "mpc" && new_mode != "open_loop") {
                    result.successful = false;
                    result.reason = "mode must be 'mpc' or 'open_loop'";
                    return result;
                }
                profile_changed = true;
            } else if (p.get_name() == "n_replay") {
                new_n_replay = p.as_int();
                if (new_n_replay < 1) {
                    result.successful = false;
                    result.reason = "n_replay must be >= 1";
                    return result;
                }
                profile_changed = true;
            } else if (p.get_name() == "open_loop_abort_on_divergence") {
                open_loop_abort_on_divergence_ = p.as_bool();
                profile_changed = true;
            } else if (p.get_name() == "open_loop_abort_max_z_error_m") {
                open_loop_abort_max_z_error_m_ = p.as_double();
                profile_changed = true;
            } else if (p.get_name() == "open_loop_abort_max_vz_error_mps") {
                open_loop_abort_max_vz_error_mps_ = p.as_double();
                profile_changed = true;
            } else if (p.get_name() == "hover_target_x") {
                new_target.x() = p.as_double();
                profile_changed = true;
            } else if (p.get_name() == "hover_target_y") {
                new_target.y() = p.as_double();
                profile_changed = true;
            } else if (p.get_name() == "hover_target_z") {
                new_target.z() = p.as_double();
                profile_changed = true;
            } else if (p.get_name() == "command_seq") {
                new_command_seq = p.as_int();
            }
        }

        if (profile_changed) {
            if (new_ocp_type == "tracking_circle_target" && new_mode == "mpc") {
                result.successful = false;
                result.reason = "tracking_circle_target supports open_loop only";
                return result;
            }

            ocp_type_ = new_ocp_type;
            mode_ = new_mode;
            n_replay_ = new_n_replay;

            if (ocp_type_changed || !is_configured_) {
                const auto& desc = OCPRegistry::getDescriptor(ocp_type_);
                ocp_dt_ = desc.dt;
                mass_kg_ = desc.default_mass_kg;
                command_mode_ = desc.command_mode;
                state_dim_ = desc.state_dim;
                control_dim_ = desc.control_dim;
                custom_make_hover_state_ = desc.make_hover_state;
                applyOcpDroneOdomMode(true);
                if (n_replay_ == 0) {
                    n_replay_ = desc.default_n_replay;
                }
            }

            setTerminalTarget(new_target);
            rebuildMpcSolver();

            bool was_unconfigured = !is_configured_;
            is_configured_ = !ocp_type_.empty() && !mode_.empty();

            if (was_unconfigured && is_configured_) {
                createModeTimers();
            }

            resetForNewCommand();
            command_paused_.store(true);
            last_command_seq_ = 0;  // Reset so command_seq:=1 always works

            RCLCPP_INFO(this->get_logger(),
                "Updated command profile: ocp=%s mode=%s n_replay=%d target=[%.3f,%.3f,%.3f] mass=%.4f cmd_mode=%s thrust=crazysim-sysid",
                ocp_type_.c_str(), mode_.c_str(), n_replay_,
                hover_target_.x(), hover_target_.y(), hover_target_.z(), mass_kg_,
                commandModeName(command_mode_));
            ensureLoggingInitialized();
            RCLCPP_INFO(this->get_logger(),
                "Profile applied. Set command_seq to start this command.");
        }

        if (new_command_seq > last_command_seq_) {
            if (!is_configured_) {
                RCLCPP_ERROR(this->get_logger(),
                    "Cannot start execution: ocp_type and mode must be configured first.");
                result.successful = false;
                result.reason = "planner not configured";
                return result;
            }
            if (ocp_type_ == "tracking_circle_target" && mode_ == "mpc") {
                result.successful = false;
                result.reason = "tracking_circle_target supports open_loop only";
                return result;
            }
            last_command_seq_ = new_command_seq;
            // Capture current drone position as the pre-solve hover target.
            // mpcReplayTick publishes this hold while !is_primed_ (solver working on first plan).
            if (hasState() && is_configured_ && mass_kg_ > 0.0) {
                const Eigen::VectorXd x_now = convertStateToAbsoluteFrame(getCurrentState());
                paused_hover_state_ = hover_controller::makeHoverState(
                    x_now, state_dim_, custom_make_hover_state_);
                maintain_hover_hold_ = true;
            } else {
                maintain_hover_hold_ = false;
            }
            command_paused_.store(false);
            ocp_active_.store(true);
            resetForNewCommand();
            RCLCPP_INFO(this->get_logger(),
                "Accepted new command_seq=%d. Starting OCP=%s mode=%s n_replay=%d.",
                last_command_seq_, ocp_type_.c_str(), mode_.c_str(), n_replay_);
        }

        return result;
    }

    void solverLoop() {
        if (mode_ != "mpc") {
            return;
        }
        if (command_paused_.load()) {
            return;
        }
        if (ocp_type_ == "tracking_circle_target") {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "tracking_circle_target is open_loop-only; rejecting MPC solve loop");
            return;
        }

        if (!hasState()) {
            const auto dbg = state_monitor_.getStateDebugSnapshot(platform_);
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for state... platform=%s has_state=%s pose=%s odom=%s relative_mode=%s odom_topic='%s'",
                platform_.c_str(),
                dbg.has_state ? "true" : "false",
                dbg.cf_pose_received ? "true" : "false",
                dbg.cf_odom_received ? "true" : "false",
                drone_state_is_relative_ ? "true" : "false",
                drone_odom_topic_.c_str());
            return;
        }

        if (!hasFreshTargetState()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for fresh target state...");
            return;
        }

        Eigen::VectorXd x0_abs = getCurrentState();
        TargetSnapshot target_snapshot = getTargetSnapshot();

        if (debug_body_relative_trace_ && drone_state_is_relative_ && x0_abs.size() >= state_dim_ && state_dim_ >= 13) {
            const Eigen::Vector4d q = x0_abs.segment<4>(6);
            const double qn = q.norm();
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "[BodyRelTrace][solver] x0 pT=(%.3f,%.3f,%.3f) vd=(%.3f,%.3f,%.3f) q=(%.4f,%.4f,%.4f,%.4f)|norm=%.4f w=(%.4f,%.4f,%.4f)",
                x0_abs(0), x0_abs(1), x0_abs(2),
                x0_abs(3), x0_abs(4), x0_abs(5),
                q(0), q(1), q(2), q(3), qn,
                x0_abs(10), x0_abs(11), x0_abs(12));
        }

        auto desc = OCPRegistry::getDescriptor(ocp_type_);

        if (!is_flying_) {
            is_flying_ = true;
            RCLCPP_INFO(this->get_logger(),
                "State received - starting RH MPC (%s)", solver_type_.c_str());
            if (platform_ == "mavros") {
                platform::mavros::markArmed(mavros_handles_);
            }
            ensureLoggingInitialized();
        }

        auto core_result = planner_core_.trySolve(makeCoreInput(x0_abs, target_snapshot));
        if (!core_result.diagnostics.empty() &&
            core_result.rejection_reason != "waiting_for_handoff_time" &&
            core_result.rejection_reason != "waiting_for_pending_handoff" &&
            core_result.rejection_reason != "terminal_freeze_active") {
            logCoreDiagnostics(core_result.diagnostics);
        }
        logSolverEvents(core_result);
        if (!core_result.solve_accepted) {
            if (core_result.terminal_freeze_engaged) {
                terminal_freeze_.store(true);
                dispatchLandingShutdownOnce("terminal freeze engaged");
            }
            if (core_result.terminal_freeze_released) {
                terminal_freeze_.store(false);
            }
            if (core_result.rejection_reason == "terminal_freeze_active") {
                dispatchLandingShutdownOnce("terminal freeze active");
            }
            if (rejectedSolveCompletesLanding(core_result, x0_abs, target_snapshot)) {
                terminal_freeze_.store(true);
                RCLCPP_WARN(this->get_logger(),
                    "Landing solve rejected near terminal by constraint_error; treating as landing completion.");
                dispatchLandingShutdownOnce("near-terminal constraint_error");
            }
            return;
        }

        stale_warning_count_ = 0;
        replay_ticks_since_solve_.store(0);
        last_replan_delay_sec_ = core_result.replan_delay_sec;

        RCLCPP_INFO(this->get_logger(),
            "[RH %d] %.1fms iters=%d x0=[%.3f,%.3f,%.3f] source=%s handoff_err(p=%.3f,v=%.3f) active=%.3fs activation=%.3fs activation_wall=%.3fs lead=%.3fs late=%.3fs replan_delay=%.3fs",
            core_result.solve_num,
            core_result.solve_result.solve_time_ms,
            core_result.solve_result.solve_iters,
            core_result.solve_result.state_trajectory.front().size() > 0 ? core_result.solve_result.state_trajectory.front()(0) : 0.0,
            core_result.solve_result.state_trajectory.front().size() > 1 ? core_result.solve_result.state_trajectory.front()(1) : 0.0,
            core_result.solve_result.state_trajectory.front().size() > 2 ? core_result.solve_result.state_trajectory.front()(2) : 0.0,
            core_result.x0_source.c_str(),
            core_result.handoff_pos_err,
            core_result.handoff_vel_err,
            core_result.active_elapsed_now_sec,
            core_result.activation_elapsed_sec,
            core_result.activation_wall_time_sec,
            core_result.solve_lead_sec,
            core_result.solve_finish_late_by_sec,
            last_replan_delay_sec_);

        is_primed_.store(true);
        terminal_freeze_.store(false);
        solve_count_ = planner_core_.solveCount();

        publishTrajectory(core_result.path.states);

        if (logging_enabled_ && logging_initialized_) {
            auto meta = toRosLogMeta(core_result.solve_log, core_result.solve_result);
            logger_.logSolveTrajectory(core_result.solve_result.state_trajectory,
                                       core_result.solve_result.control_trajectory,
                                       meta);
        }
    }

    void mpcReplayTick() {
        if (mode_ != "mpc") {
            return;
        }
        if (command_paused_.load()) {
            publishPausedHoverHoldTick();
            return;
        }
        if (!is_primed_.load()) {
            return;  // high-level goto keeps drone in place until first solve arrives
        }

        last_target_snapshot_ = getTargetSnapshot();
        auto replay = planner_core_.sampleReplay(Clock::now(), last_target_snapshot_);
        if (!replay.has_replay_sample) {
            return;
        }
        logCoreDiagnostics(replay.diagnostics);
        logReplaySolverEvents(replay);

        replay_ticks_since_solve_.fetch_add(1);

        if (replay.stale) {
            ++stale_warning_count_;
            RCLCPP_WARN(this->get_logger(),
                "Trajectory stale, clamping to terminal");

            if (stale_warning_count_ >= 3) {
                holdHoverAndPause("trajectory stale 3x");
                return;
            }
        } else {
            stale_warning_count_ = 0;
        }

        publishCommand(replay.command);
        publishDebugMarkers(replay.command, replay.diagnostics);
    }

    bool validateTrajectory(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) {
        if (X.size() < 2) return false;
        const auto desc = OCPRegistry::getDescriptor(ocp_type_);
        
        // Physical bounds checking
        for (size_t k = 0; k < X.size(); ++k) {
            if (X[k].size() < state_dim_) continue;
            // Altitude check (reject negative altitude)
            if (!desc.skip_altitude_validation && X[k].size() >= 3 && X[k](2) < -0.05) return false;
            
            // Velocity check (reject > 20 m/s)
            if (X[k].size() >= 6 && X[k].segment(3, 3).norm() > 20.0) return false;
            
            // Angular rate check (reject > 50 rad/s)
            if (X[k].size() >= 13 && X[k].segment(10, 3).norm() > 50.0) return false;
        }
        
        if (!U.empty()) {
            for (size_t k = 0; k < U.size(); ++k) {
                // Thrust bounds (FMAX is ~0.6-0.7 for crazyflie, let's say max 10.0 to be safe across platforms)
                if (U[k].size() > 0 && (U[k](0) < -0.1 || U[k](0) > 50.0)) {
                    return false;
                }
            }
        }
        
        return true;
    }

    SolverResult callSolver(const Eigen::VectorXd& state,
                            const TargetSnapshot& target_snapshot = TargetSnapshot{}) {
        SolverResult result;
        if (solver_type_ == "alipddp" && alipddp_mpc_) {
            double t_abs = this->now().seconds();
            auto desc = OCPRegistry::getDescriptor(ocp_type_);
            std::any extra;
            if (desc.prepare_extra) {
                // Make the latest buffer available to OCP-specific prepare_extra
                // callbacks; they decide whether to use it or fall back.
                runtime_cfg_.target_accel_buffer = state_monitor_.getTargetAccelBuffer();
                extra = desc.prepare_extra(toPlannerConfig(runtime_cfg_), t_abs, target_snapshot);
            }

            const Eigen::Vector3d target_accel =
                target_snapshot.valid ? target_snapshot.acceleration : Eigen::Vector3d::Zero();
            auto r = alipddp_mpc_->solve(state, target_accel, extra, t_abs);
            
            result.success = r.success;
            result.next_state = r.next_state;
            result.state_trajectory = r.state_trajectory;
            result.control_trajectory = r.control_trajectory;
            result.solve_time_ms = r.solve_time_ms;
            result.constraint_error = r.constraint_error;
            result.solve_iters = r.solve_iters;
            result.solve_start_time = r.solve_start_time;
            result.solve_finish_time = r.solve_finish_time;
            result.solve_timestamp = r.solve_timestamp;
            result.extra = r.extra_params;

            result.target_snapshot_pos = target_snapshot.position;
            result.target_snapshot_vel = target_snapshot.velocity;
            result.target_snapshot_acc = target_snapshot.acceleration;
            result.target_snapshot_quat = target_snapshot.orientation;
            result.target_snapshot_omega = target_snapshot.angular_velocity;
            result.target_snapshot_beta = target_snapshot.angular_acceleration;
            result.target_motion_source = "snapshot";
        }
        return result;
    }

    void publishCommand(const Eigen::VectorXd& s, const Eigen::VectorXd& u) {
        const auto& desc = OCPRegistry::getDescriptor(ocp_type_);
        const auto cmd = planner_core::makePlannerCommand(
            corePlatform(), desc, s, u, last_target_snapshot_);
        publishCommand(cmd);
    }

    void publishCommand(const planner_core::PlannerCommand& cmd) {
        if (landing_shutdown_sent_.load()) {
            return;
        }
        if (cmd.kind == planner_core::CommandKind::Hover && platform_ == "crazyflie") {
            platform::crazyflie::publishHoverCommandDirect(cf_handles_, hoverArray(cmd.hover));
        } else if (cmd.kind == planner_core::CommandKind::BodyRate && platform_ == "mavros") {
            platform::mavros::publishBodyRateCommand(mavros_handles_, cmd.control);
        } else if (cmd.kind == planner_core::CommandKind::FullState && platform_ == "crazyflie") {
            platform::crazyflie::publishCommand(this, cf_handles_, cmd.state, cmd.control, mass_kg_);
        } else if (cmd.kind == planner_core::CommandKind::FullState && platform_ == "mavros") {
            platform::mavros::publishFullStateCommand(mavros_handles_, cmd.state);
        }
    }

    void publishTrajectory(const std::vector<Eigen::VectorXd>& traj) {
        nav_msgs::msg::Path path;
        path.header.stamp = this->now();
        path.header.frame_id = "world";
        for (const auto& s : traj) {
            if (s.size() < 10) {
                continue;
            }
            geometry_msgs::msg::PoseStamped p;
            p.header.frame_id = "world";
            p.pose.position.x = s(0);
            p.pose.position.y = s(1);
            p.pose.position.z = s(2);
            p.pose.orientation.w = s(6);
            p.pose.orientation.x = s(7);
            p.pose.orientation.y = s(8);
            p.pose.orientation.z = s(9);
            path.poses.push_back(p);
        }
        traj_pub_->publish(path);
    }

    void openLoopStartupCheck() {
        if (command_paused_.load()) {
            if (open_loop_phase_ == OpenLoopPhase::Hold) {
                publishPausedHoverHoldTick();
            }
            return;
        }
        if (!hasState()) {
            const auto dbg = state_monitor_.getStateDebugSnapshot(platform_);
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[OpenLoop] Waiting for state... platform=%s has_state=%s pose=%s odom=%s relative_mode=%s odom_topic='%s'",
                platform_.c_str(),
                dbg.has_state ? "true" : "false",
                dbg.cf_pose_received ? "true" : "false",
                dbg.cf_odom_received ? "true" : "false",
                drone_state_is_relative_ ? "true" : "false",
                drone_odom_topic_.c_str());
            return;
        }

        auto desc = OCPRegistry::getDescriptor(ocp_type_);
        TargetSnapshot target_snapshot = getTargetSnapshot();
        const bool target_valid_for_ocp = isTargetValidForDescriptor(desc, target_snapshot);

        if (desc.needs_target_trajectory || desc.validate_target) {
            if (!target_valid_for_ocp) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "[OpenLoop] IDLE — waiting for valid target state...");
                publishPausedHoverHoldTick();
                return;
            }
        }

        if (desc.needs_target_trajectory) {
            if (!state_monitor_.hasTargetTrajectory(this->now())) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "[OpenLoop] IDLE — waiting for /target/predicted_accel...");
                publishPausedHoverHoldTick();
                return;
            }
        }

        startup_timer_->cancel();

        Eigen::VectorXd x0 = getCurrentState();
        if (desc.transform_state) {
            x0 = desc.transform_state(x0, target_snapshot);
        }
        RCLCPP_INFO(this->get_logger(), "[OpenLoop] Solving...");
        SolverResult result = callSolver(x0, target_snapshot);

        if (desc.post_process_result) {
            desc.post_process_result(result, target_snapshot);
        }

        if (ocp_type_ == "tracking_circle_target" && result.is_relative_plan) {
            TrackingCircleTargetOCP::TrackingCircleTargetExtra ex;
            if (!tryGetTrackingCircleTargetExtra(result.extra, ex)) {
                RCLCPP_ERROR(this->get_logger(),
                    "[OpenLoop] FAILED — tracking_circle_target missing predicted acceleration buffer");
                return;
            }

            // Canonical interface reconstruction path used for replay, path
            // publishing, and solve logging.
            std::vector<Eigen::Vector3d> tgt_pos_nodes;
            std::vector<Eigen::Vector3d> tgt_vel_nodes;
            result.state_trajectory = reconstructTrackingCircleTargetAbsoluteTrajectory(
                result.state_trajectory,
                result.target_snapshot_pos,
                result.target_snapshot_vel,
                ex.buf,
                ex.t_abs,
                &tgt_pos_nodes,
                &tgt_vel_nodes);
            result.target_world_pos_trajectory = std::move(tgt_pos_nodes);
            result.target_world_vel_trajectory = std::move(tgt_vel_nodes);
            result.target_motion_source = "odom+predicted_accel";
            result.is_relative_plan = false;
        }

        if (!result.success || result.state_trajectory.size() < 2) {
            RCLCPP_ERROR(this->get_logger(), "[OpenLoop] FAILED");
            return;
        }

        ol_ref_X_ = result.state_trajectory;
        ol_ref_U_ = result.control_trajectory;

        if (logging_enabled_) {
            if (!logging_initialized_) {
                auto dec = OCPRegistry::getDescriptor(ocp_type_);
                logging_initialized_ = logger_.initialize(
                    drone_name_, ocp_type_, mode_, solver_type_, this->get_logger(), mass_kg_,
                    state_dim_, dec.state_names, dec.control_names, dec.log_state_headers,
                    command_mode_, dec.extract_actual_state_row);
            }
            if (logging_initialized_) {
                planner_logging::SolveLogMeta meta;
                meta.solve_num = 0;
                meta.solve_time_ms = result.solve_time_ms;
                meta.solve_iters = result.solve_iters;
                meta.is_relative_plan = result.is_relative_plan;
                meta.ocp_dt = ocp_dt_;
                meta.target_snapshot_pos = result.target_snapshot_pos;
                meta.target_snapshot_vel = result.target_snapshot_vel;
                meta.target_snapshot_acc = result.target_snapshot_acc;
                meta.target_snapshot_quat = result.target_snapshot_quat;
                meta.target_snapshot_omega = result.target_snapshot_omega;
                meta.target_snapshot_beta = result.target_snapshot_beta;
                meta.target_world_pos_trajectory = result.target_world_pos_trajectory;
                meta.target_world_vel_trajectory = result.target_world_vel_trajectory;

                auto desc = OCPRegistry::getDescriptor(ocp_type_);
                if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative) {
                    meta.coord_mode = "body_relative";
                } else if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative) {
                    meta.coord_mode = "target_frame_relative";
                } else if (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::AbsoluteShiftedTarget) {
                    meta.coord_mode = "absolute_relative";
                } else {
                    meta.coord_mode = "absolute";
                }
                if (desc.prepare_log_meta) {
                    desc.prepare_log_meta(meta, result.extra, runtime_cfg_);
                }

                logger_.logSolveTrajectory(ol_ref_X_, ol_ref_U_, meta);
                planner_logging::SolverEventLogRow row;
                row.solve_num = 0;
                row.event = "accepted";
                row.reason = "open_loop";
                row.coord_mode = meta.coord_mode;
                row.solve_time_ms = result.solve_time_ms;
                row.solve_iters = static_cast<double>(result.solve_iters);
                row.constraint_error = result.constraint_error;
                logger_.logSolverEvent(row);
            }
        }
        publishTrajectory(makeTrajectoryForPublishing(result));

        ol_replay_step_ = 0;
        ol_replay_start_time_ = Clock::now();
        setOpenLoopPhase(OpenLoopPhase::Replay);
        stopOpenLoopHoldTimer();
        stopReplayTimer();
        const int period_ms = static_cast<int>(std::round(ocp_dt_ * 1000.0));
        replay_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(period_ms),
            std::bind(&PlannerNode::openLoopReplayTick, this),
            solver_cb_group_);

        RCLCPP_INFO(this->get_logger(), "[OpenLoop] Replaying %zu steps", ol_ref_X_.size());
    }

    void openLoopReplayTick() {
        if (open_loop_phase_ != OpenLoopPhase::Replay) {
            return;
        }
        if (command_paused_.load()) {
            return;
        }

        const int N = static_cast<int>(ol_ref_X_.size()) - 1;

        if (ol_replay_step_ >= N) {
            if (shouldDisarmOnLandingFinish()) {
                if (dispatchLandingShutdownOnce("open-loop landing complete")) {
                    stopReplayTimer();
                    setOpenLoopPhase(OpenLoopPhase::Idle);
                    ocp_active_.store(false);
                    RCLCPP_INFO(this->get_logger(),
                        "[OpenLoop] Landing replay complete. Crazyflie shutdown dispatched.");
                }
                return;
            }
            if (hasState() && is_configured_) {
                const Eigen::VectorXd x_now = convertStateToAbsoluteFrame(getCurrentState());
                paused_hover_state_ = hover_controller::makeHoverState(
                    x_now, state_dim_, custom_make_hover_state_);
                maintain_hover_hold_ = true;
            }
            transitionOpenLoopToHold("nominal horizon end");
            return;
        }

        const int step = ol_replay_step_;
        const Eigen::VectorXd& x_cmd = ol_ref_X_[step];
        const Eigen::VectorXd u_cmd = (step < static_cast<int>(ol_ref_U_.size()))
            ? ol_ref_U_[step] : Eigen::VectorXd::Zero(4);

        // Divergence check compares world-frame z — meaningless in body-relative mode
        // (x(2) is p_T_body[2], not drone altitude).  Skip the check entirely.
        if (!drone_state_is_relative_ && open_loop_abort_on_divergence_ &&
            x_cmd.size() >= 6 && hasState()) {
            const Eigen::VectorXd x_act = convertStateToAbsoluteFrame(getCurrentState());
            if (x_act.size() >= 6) {
                const double z_err = std::abs(x_act(2) - x_cmd(2));
                const double vz_err = std::abs(x_act(5) - x_cmd(5));
                if (z_err > open_loop_abort_max_z_error_m_ ||
                    vz_err > open_loop_abort_max_vz_error_mps_) {
                    std::ostringstream ss;
                    ss << "divergence at step=" << step
                       << " z_err=" << z_err << " (thr=" << open_loop_abort_max_z_error_m_ << ")"
                       << " vz_err=" << vz_err << " (thr=" << open_loop_abort_max_vz_error_mps_ << ")";
                    abortOpenLoopReplay(ss.str());
                    return;
                }
            }
        }

        last_target_snapshot_ = getTargetSnapshot();
        publishCommand(x_cmd, u_cmd);

        ++ol_replay_step_;
    }

    void openLoopHoldTick() {
        if (mode_ != "open_loop" || open_loop_phase_ != OpenLoopPhase::Hold) {
            return;
        }
        publishPausedHoverHoldTick();
    }

    std::string ocp_type_, platform_, solver_type_, mode_, drone_name_;
    bool drone_state_is_relative_ = false;
    OCPDescriptor::DroneOdomMode drone_odom_mode_ = OCPDescriptor::DroneOdomMode::Absolute;
    std::string drone_odom_topic_;
    std::string body_relative_odom_topic_ = "/drone/body_relative_odom";
    std::string target_frame_odom_topic_ = "/drone/target_frame_odom";
    std::string target_odom_topic_ = "/target/odom";
    std::string target_accel_topic_ = "/target/accel";
    std::string target_predicted_accel_topic_ = "/target/predicted_accel";
    bool debug_body_relative_trace_ = false;

    bool enable_terminal_freeze_ = true;
    double terminal_freeze_enter_pos_ = 0.20;
    double terminal_freeze_enter_vel_ = 0.10;
    bool terminal_freeze_require_vel_ = false;
    double terminal_freeze_exit_pos_ = 0.20;

    bool logging_enabled_;
    double ocp_dt_ = 0.05;
    PlannerRuntimeConfig runtime_cfg_;
    double mass_kg_ = 0.027;
    bool open_loop_abort_on_divergence_ = false;
    double open_loop_abort_max_z_error_m_ = 0.50;
    double open_loop_abort_max_vz_error_mps_ = 1.00;
    int n_replay_ = 4;
    double max_constraint_error_ = 1.0;
    bool skip_trajectory_validation_ = false;
    double max_first_solve_age_sec_ = 1.0;
    double max_relative_position_norm_ = 10.0;
    double max_relative_vertical_abs_ = 5.0;
    double max_relative_velocity_norm_ = 8.0;
    OCPDescriptor::CommandMode command_mode_ = OCPDescriptor::CommandMode::CmdFullState;
    int state_dim_ = 13;
    int control_dim_ = 4;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> custom_make_hover_state_ = nullptr;

    Eigen::Vector3d hover_target_ = Eigen::Vector3d::Zero();
    Eigen::VectorXd terminal_state_ = Eigen::VectorXd();

    std::unique_ptr<QuadrotorMPC> alipddp_mpc_;

    platform::crazyflie::Handles cf_handles_;
    platform::mavros::Handles mavros_handles_;
    platform::target_tracker::Handles target_handles_;

    float hover_thrust_param_ = 0.3f;
    TargetSnapshot last_target_snapshot_;

    rclcpp::TimerBase::SharedPtr solver_timer_;
    rclcpp::TimerBase::SharedPtr mpc_replay_timer_;
    rclcpp::TimerBase::SharedPtr startup_timer_;
    rclcpp::TimerBase::SharedPtr replay_timer_;
    rclcpp::TimerBase::SharedPtr open_loop_hold_timer_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_marker_pub_;

    planner_core::PlannerCore planner_core_;
    bool planner_core_configured_ = false;
    std::deque<Eigen::Vector3d> target_history_;

    StateMonitor state_monitor_;

    rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
    rclcpp::CallbackGroup::SharedPtr solver_cb_group_;
    rclcpp::CallbackGroup::SharedPtr replay_cb_group_;

    bool is_flying_ = false;
    int solve_count_ = 0;
    std::atomic<int>  replay_ticks_since_solve_{0};
    std::atomic<bool> is_primed_{false};
    std::atomic<bool> terminal_freeze_{false};
    std::atomic<bool> landing_shutdown_sent_{false};
    Eigen::Vector3d terminal_position_abs_ = Eigen::Vector3d::Zero();
    Clock::time_point last_accepted_solve_timestamp_{};
    double last_replan_delay_sec_ = 0.0;

    std::vector<Eigen::VectorXd> ol_ref_X_, ol_ref_U_;
    int ol_replay_step_ = 0;
    bool ol_done_logged_ = false;
    Clock::time_point ol_replay_start_time_;
    OpenLoopPhase open_loop_phase_ = OpenLoopPhase::Idle;

    bool logging_initialized_ = false;
    planner_logging::CsvLogger logger_;

    std::atomic<bool> command_paused_{true};
    int last_command_seq_ = 0;
    int stale_warning_count_ = 0;
    bool maintain_hover_hold_ = false;
    Eigen::VectorXd paused_hover_state_ = Eigen::VectorXd();
    std::mutex command_mutex_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

    std::atomic<bool> ocp_active_{false};
    bool is_configured_ = true;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
