/// @file planner_node.cpp
/// @brief CoManDO planner node - platform-agnostic MPC planner.

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <nav_msgs/msg/path.hpp>
#include <Eigen/Dense>

#include "quadrotor_mpc.hpp"
#include "ocp_registry.hpp"
#include "platform/crazyflie.hpp"
#include "platform/px4.hpp"
#include "platform/target_tracker.hpp"
#include "state_monitor.hpp"
#include "trajectory_replayer.hpp"
#include "planner_runtime_config.hpp"
#include "planner_logging.hpp"
#include "hover_controller.hpp"

#include <chrono>
#include <memory>
#include <cmath>
#include <mutex>
#include <atomic>
#include <vector>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("comando_planner") {
        const PlannerRuntimeConfig runtime_cfg = loadPlannerRuntimeConfig(this);
        runtime_cfg_ = runtime_cfg;

        ocp_type_ = runtime_cfg.ocp_type;
        drone_name_ = runtime_cfg.drone_name;
        logging_enabled_ = runtime_cfg.enable_logging;
        platform_ = runtime_cfg.platform;
        solver_type_ = runtime_cfg.solver;
        mode_ = runtime_cfg.mode;
        n_replay_ = runtime_cfg.n_replay;
        mass_kg_ = runtime_cfg.mass_kg;
        target_odom_topic_ = runtime_cfg.target_odom_topic;
        target_accel_topic_ = runtime_cfg.target_accel_topic;
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
            setTerminalTarget(hover_target_);
            rebuildMpcSolver();
            replay_ticks_since_solve_.store(n_replay_);
            is_configured_ = true;
        } else {
            ocp_dt_ = 0.0;
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
                this, sensor_cb_group_, drone_name_,
                state_monitor_.crazyflieState(), state_monitor_.stateMutex(), cf_handles_);
        } else if (platform_ == "px4") {
            platform::px4::setup(
                this, sensor_cb_group_,
                state_monitor_.px4State(), state_monitor_.stateMutex(), px4_handles_);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unknown platform: %s", platform_.c_str());
            throw std::runtime_error("Unknown platform: " + platform_);
        }

        // Keep target subscriptions alive regardless of active OCP type,
        // so runtime OCP switching to stateswitch works without restart.
        platform::target_tracker::setup(
            this, sensor_cb_group_, target_odom_topic_, target_accel_topic_,
            state_monitor_.targetState(), state_monitor_.targetMutex(), target_handles_);

        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);

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
                "Command profile target: [%.3f, %.3f, %.3f] mass=%.4f",
                hover_target_.x(), hover_target_.y(), hover_target_.z(), mass_kg_);
        } else {
            RCLCPP_WARN(this->get_logger(),
                "Planner started UNCONFIGURED. Use ocp_launch.py to set ocp_type and mode.");
        }
        RCLCPP_INFO(this->get_logger(),
            "To run next OCP from another terminal: set ocp/mode/n_replay/target then increment command_seq.");
    }

private:
    bool hasState() const {
        return state_monitor_.hasState(platform_);
    }

    bool hasFreshTargetState() const {
        if (ocp_type_.empty()) return true;
        bool needs_target = OCPRegistry::getDescriptor(ocp_type_).validate_target != nullptr;
        return state_monitor_.hasFreshTargetState(needs_target, this->now());
    }

    TargetSnapshot getTargetSnapshot() const {
        if (ocp_type_.empty()) return TargetSnapshot{true};
        bool needs_target = OCPRegistry::getDescriptor(ocp_type_).validate_target != nullptr;
        return state_monitor_.getTargetSnapshot(needs_target, this->now());
    }

    Eigen::VectorXd getCurrentState() const {
        return state_monitor_.getCurrentState(platform_);
    }

    void setTerminalTarget(const Eigen::Vector3d& xyz) {
        hover_target_ = xyz;
        terminal_position_abs_ = xyz;

        Eigen::VectorXd terminal = Eigen::VectorXd::Zero(13);
        terminal(0) = xyz.x();
        terminal(1) = xyz.y();
        terminal(2) = xyz.z();
        terminal(6) = 1.0;
        terminal_state_ = terminal;

        if (alipddp_mpc_) {
            alipddp_mpc_->setTerminalState(terminal_state_);
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
        trajectory_replayer_.clear();
        replay_ticks_since_solve_.store(n_replay_);
        is_primed_.store(false);
        terminal_freeze_.store(false);
        stale_warning_count_ = 0;
    }

    void publishPausedHoverHoldTick() {
        hover_controller::doHoverHoldTick(
            is_configured_,
            mass_kg_,
            maintain_hover_hold_,
            hasState(),
            getCurrentState(),
            paused_hover_state_,
            [this](const auto& x, const auto& u) { publishCommand(x, u); }
        );
        // Logging is maintained if needed
            // Only log if OCP is active
            if (ocp_active_.load()) {
                const Eigen::VectorXd x_now = getCurrentState();
                Eigen::VectorXd x_hold = (paused_hover_state_.size() >= 13)
                    ? paused_hover_state_
                    : hover_controller::makeHoverState(x_now);
                Eigen::VectorXd u_hover = hover_controller::makeHoverControl(mass_kg_);
                logger_.logActualState(x_now, -1);
                logger_.logCommandedState(x_hold, u_hover, -1);
            }
    }

    void holdHoverAndPause(const std::string& reason) {
        hover_controller::enterHoverHold(
            reason,
            is_configured_,
            mass_kg_,
            hasState(),
            getCurrentState(),
            command_paused_,
            maintain_hover_hold_,
            paused_hover_state_,
            [this](const auto& x, const auto& u) { publishCommand(x, u); },
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
        bool mode_changed = false;

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
                mode_changed = true;
            } else if (p.get_name() == "n_replay") {
                new_n_replay = p.as_int();
                if (new_n_replay < 1) {
                    result.successful = false;
                    result.reason = "n_replay must be >= 1";
                    return result;
                }
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
            ocp_type_ = new_ocp_type;
            mode_ = new_mode;
            n_replay_ = new_n_replay;

            if (ocp_type_changed || !is_configured_) {
                ocp_dt_ = OCPRegistry::getDT(ocp_type_);
                mass_kg_ = OCPRegistry::getDefaultMassKg(ocp_type_);
                if (n_replay_ == 0) {
                    n_replay_ = OCPRegistry::getDefaultNReplay(ocp_type_);
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
                "Updated command profile: ocp=%s mode=%s n_replay=%d target=[%.3f,%.3f,%.3f] mass=%.4f",
                ocp_type_.c_str(), mode_.c_str(), n_replay_,
                hover_target_.x(), hover_target_.y(), hover_target_.z(), mass_kg_);
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
            last_command_seq_ = new_command_seq;
            maintain_hover_hold_ = false;
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

        if (!hasState()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for state...");
            return;
        }

        if (!hasFreshTargetState()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for fresh target state...");
            return;
        }

        if (replay_ticks_since_solve_.load() < n_replay_) {
            return;
        }

        Eigen::VectorXd x0_abs = getCurrentState();
        TargetSnapshot target_snapshot = getTargetSnapshot();

        Eigen::VectorXd x0 = x0_abs;
        auto desc = OCPRegistry::getDescriptor(ocp_type_);
        if (desc.transform_state) {
            x0 = desc.transform_state(x0_abs, target_snapshot);
        }

        const double pos_err = (desc.transform_state != nullptr)
            ? x0.segment(0, 3).norm()
            : (x0_abs.segment(0, 3) - terminal_position_abs_).norm();
        const double vel_err = (desc.transform_state != nullptr)
            ? x0.segment(3, 3).norm()
            : x0_abs.segment(3, 3).norm();

        if (terminal_freeze_.load()) {
            if (pos_err > terminal_freeze_exit_pos_) {
                terminal_freeze_.store(false);
                RCLCPP_INFO(this->get_logger(),
                    "Terminal freeze released: pos_err=%.3f", pos_err);
            } else {
                return;
            }
        }

        const bool freeze_condition = (pos_err < terminal_freeze_enter_pos_) &&
            (!terminal_freeze_require_vel_ || vel_err < terminal_freeze_enter_vel_);
        if (enable_terminal_freeze_ && is_primed_.load() && freeze_condition) {
            terminal_freeze_.store(true);
            RCLCPP_INFO(this->get_logger(),
                "Terminal freeze engaged: pos_err=%.3f vel_err=%.3f", pos_err, vel_err);
            return;
        }

        if (!is_flying_) {
            is_flying_ = true;
            RCLCPP_INFO(this->get_logger(),
                "State received - starting RH MPC (%s)", solver_type_.c_str());
            if (platform_ == "px4") {
                platform::px4::arm(this, px4_handles_);
            }
            if (logging_enabled_ && !logging_initialized_) {
                logging_initialized_ = logger_.initialize(
                    drone_name_, ocp_type_, mode_, solver_type_, this->get_logger(), mass_kg_);
            }
        RCLCPP_INFO(this->get_logger(),
            "x0=[%.3f,%.3f,%.3f | %.3f,%.3f,%.3f]",
            x0(0), x0(1), x0(2), x0(3), x0(4), x0(5));
    }

        const Eigen::Vector3d target_accel = target_snapshot.acceleration;

        SolverResult result = callSolver(x0, target_accel);
        if (desc.post_process_result) {
            desc.post_process_result(result, target_snapshot);
        }

        if (!result.success || result.state_trajectory.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "Solve FAILED - holding");
            Eigen::VectorXd x_hold = x0_abs;
            x_hold.segment(3, 3).setZero();
            x_hold.segment(10, 3).setZero();
            publishCommand(x_hold, Eigen::VectorXd::Zero(4));
            return;
        }

        stale_warning_count_ = 0;

        RCLCPP_INFO(this->get_logger(),
            "[RH %d] %.1fms iters=%d x0=[%.3f,%.3f,%.3f]",
            solve_count_, result.solve_time_ms, result.solve_iters,
            x0(0), x0(1), x0(2));

        trajectory_replayer_.updatePlan(
            result.state_trajectory,
            result.control_trajectory,
            result.solve_time_ms,
            solve_count_,
            result.is_relative_plan,
            result.solve_timestamp,
            ocp_dt_);

        replay_ticks_since_solve_.store(0);
        is_primed_.store(true);

        publishTrajectory(result.state_trajectory);

        if (logging_enabled_ && logging_initialized_) {
            planner_logging::SolveLogMeta meta;
            meta.solve_num = solve_count_;
            meta.solve_time_ms = result.solve_time_ms;
            meta.solve_iters = result.solve_iters;
            meta.is_relative_plan = result.is_relative_plan;
            meta.ocp_dt = ocp_dt_;
            meta.target_snapshot_pos = result.target_snapshot_pos;
            meta.target_snapshot_vel = result.target_snapshot_vel;
            meta.target_snapshot_acc = result.target_snapshot_acc;

            auto desc = OCPRegistry::getDescriptor(ocp_type_);
            if (desc.prepare_log_meta) {
                desc.prepare_log_meta(meta, result.extra, runtime_cfg_);
            }

            logger_.logSolveTrajectory(result.state_trajectory, result.control_trajectory, meta);
        }

        ++solve_count_;
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
            return;
        }

        auto replay = trajectory_replayer_.sample(Clock::now(), ocp_dt_);
        if (!replay.has_plan) {
            replay_ticks_since_solve_.fetch_add(1);
            return;
        }

        const int active_solve_num = replay.active_solve_num;
        const bool plan_is_relative = replay.plan_is_relative;
        const double elapsed = replay.elapsed;
        const double horizon_end = replay.horizon_end;

        Eigen::VectorXd x_cmd = replay.x_cmd;
        Eigen::VectorXd u_cmd = replay.u_cmd;
        const Eigen::VectorXd x_rel_k = replay.x_rel_k;

        const bool stale = (elapsed > horizon_end + 0.2);
        if (stale) {
            ++stale_warning_count_;
            if (plan_is_relative) {
                RCLCPP_WARN(this->get_logger(),
                    "Stateswitch trajectory stale (elapsed=%.3fs, horizon=%.3fs), clamping to terminal",
                    elapsed, horizon_end);
            } else {
                RCLCPP_WARN(this->get_logger(),
                    "Trajectory stale (elapsed=%.3fs, horizon=%.3fs), clamping to terminal",
                    elapsed, horizon_end);
            }

            if (stale_warning_count_ >= 3) {
                holdHoverAndPause("trajectory stale 3x");
                return;
            }
        } else {
            stale_warning_count_ = 0;
        }

        if (plan_is_relative) {
            const auto [tgt_pos_now, tgt_vel_now] = state_monitor_.getTargetPositionVelocity();
        if (x_rel_k.size() >= 13) {
                x_cmd.segment(0, 3) = x_rel_k.segment(0, 3) + tgt_pos_now;
                x_cmd.segment(3, 3) = x_rel_k.segment(3, 3) + tgt_vel_now;
                x_cmd.segment(6, 7) = x_rel_k.segment(6, 7);
            }
        }

        replay_ticks_since_solve_.fetch_add(1);
        publishCommand(x_cmd, u_cmd);

        if (logging_enabled_ && logging_initialized_) {
            const Eigen::VectorXd act = getCurrentState();
            logger_.logActualState(act, active_solve_num);
            logger_.logCommandedState(x_cmd, u_cmd, active_solve_num);
        }
    }

    SolverResult callSolver(const Eigen::VectorXd& state,
                            const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero()) {
        SolverResult result;
        if (solver_type_ == "alipddp" && alipddp_mpc_) {
            double t_abs = this->now().seconds();
            auto desc = OCPRegistry::getDescriptor(ocp_type_);
            std::any extra;
            if (desc.prepare_extra) {
                extra = desc.prepare_extra(runtime_cfg_, t_abs);
            }

            auto r = alipddp_mpc_->solve(state, target_accel, extra, t_abs);
            
            result.success = r.success;
            result.next_state = r.next_state;
            result.state_trajectory = r.state_trajectory;
            result.control_trajectory = r.control_trajectory;
            result.solve_time_ms = r.solve_time_ms;
            result.solve_iters = r.solve_iters;
            result.solve_timestamp = r.solve_timestamp;
            result.extra = r.extra_params;
        }
        return result;
    }

    void publishCommand(const Eigen::VectorXd& s, const Eigen::VectorXd& u) {
        if (platform_ == "crazyflie") {
            platform::crazyflie::publishCommand(this, cf_handles_, s, u, mass_kg_);
        } else if (platform_ == "px4") {
            platform::px4::publishCommand(this, px4_handles_, s);
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
            publishPausedHoverHoldTick();
            return;
        }
        if (!hasState()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[OpenLoop] Waiting for state...");
            return;
        }
        startup_timer_->cancel();

        Eigen::VectorXd x0 = getCurrentState();
        RCLCPP_INFO(this->get_logger(), "[OpenLoop] Solving...");
        SolverResult result = callSolver(x0);

        auto desc = OCPRegistry::getDescriptor(ocp_type_);
        if (desc.post_process_result) {
            TargetSnapshot mock_t = getTargetSnapshot();
            desc.post_process_result(result, mock_t);
        }

        if (!result.success || result.state_trajectory.size() < 2) {
            RCLCPP_ERROR(this->get_logger(), "[OpenLoop] FAILED");
            return;
        }

        ol_ref_X_ = result.state_trajectory;
        ol_ref_U_ = result.control_trajectory;

        if (logging_enabled_) {
            if (!logging_initialized_) {
                logging_initialized_ = logger_.initialize(
                    drone_name_, ocp_type_, mode_, solver_type_, this->get_logger(), mass_kg_);
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

                auto desc = OCPRegistry::getDescriptor(ocp_type_);
                if (desc.prepare_log_meta) {
                    desc.prepare_log_meta(meta, result.extra, runtime_cfg_);
                }

                logger_.logSolveTrajectory(ol_ref_X_, ol_ref_U_, meta);
            }
        }
        publishTrajectory(ol_ref_X_);

        ol_replay_step_ = 0;
        ol_replay_start_time_ = Clock::now();
        const int period_ms = static_cast<int>(std::round(ocp_dt_ * 1000.0));
        replay_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(period_ms),
            std::bind(&PlannerNode::openLoopReplayTick, this),
            solver_cb_group_);

        RCLCPP_INFO(this->get_logger(), "[OpenLoop] Replaying %zu steps", ol_ref_X_.size());
    }

    void openLoopReplayTick() {
        if (command_paused_.load()) {
            publishPausedHoverHoldTick();
            return;
        }

        const int N = static_cast<int>(ol_ref_X_.size()) - 1;

        if (ol_replay_step_ >= N) {
            holdHoverAndPause("[OpenLoop] Trajectory complete");
            return;
        }

        const int step = ol_replay_step_;
        const Eigen::VectorXd& x_cmd = ol_ref_X_[step];
        const Eigen::VectorXd u_cmd = (step < static_cast<int>(ol_ref_U_.size()))
            ? ol_ref_U_[step] : Eigen::VectorXd::Zero(4);

        publishCommand(x_cmd, u_cmd);

        if (logging_enabled_ && logging_initialized_) {
            Eigen::VectorXd act = getCurrentState();
            logger_.logActualState(act, 0);
            logger_.logCommandedState(x_cmd, u_cmd, 0);
        }

        ++ol_replay_step_;
    }

    std::string ocp_type_, platform_, solver_type_, mode_, drone_name_;
    std::string target_odom_topic_ = "/target/odom";
    std::string target_accel_topic_ = "/target/accel";

    bool enable_terminal_freeze_ = true;
    double terminal_freeze_enter_pos_ = 0.20;
    double terminal_freeze_enter_vel_ = 0.10;
    bool terminal_freeze_require_vel_ = false;
    double terminal_freeze_exit_pos_ = 0.20;

    bool logging_enabled_;
    double ocp_dt_ = 0.05;
    PlannerRuntimeConfig runtime_cfg_;
    double mass_kg_ = 0.027;
    int n_replay_ = 4;

    Eigen::Vector3d hover_target_ = Eigen::Vector3d::Zero();
    Eigen::VectorXd terminal_state_ = Eigen::VectorXd::Zero(13);

    std::unique_ptr<QuadrotorMPC> alipddp_mpc_;

    platform::crazyflie::Handles cf_handles_;
    platform::px4::Handles px4_handles_;
    platform::target_tracker::Handles target_handles_;

    rclcpp::TimerBase::SharedPtr solver_timer_;
    rclcpp::TimerBase::SharedPtr mpc_replay_timer_;
    rclcpp::TimerBase::SharedPtr startup_timer_;
    rclcpp::TimerBase::SharedPtr replay_timer_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;

    TrajectoryReplayer trajectory_replayer_;
    std::atomic<int> replay_ticks_since_solve_{0};

    StateMonitor state_monitor_;

    rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
    rclcpp::CallbackGroup::SharedPtr solver_cb_group_;
    rclcpp::CallbackGroup::SharedPtr replay_cb_group_;

    bool is_flying_ = false;
    int solve_count_ = 0;
    std::atomic<bool> is_primed_{false};
    std::atomic<bool> terminal_freeze_{false};
    Eigen::Vector3d terminal_position_abs_ = Eigen::Vector3d::Zero();

    std::vector<Eigen::VectorXd> ol_ref_X_, ol_ref_U_;
    int ol_replay_step_ = 0;
    bool ol_done_logged_ = false;
    Clock::time_point ol_replay_start_time_;

    bool logging_initialized_ = false;
    planner_logging::CsvLogger logger_;

    std::atomic<bool> command_paused_{true};
    int last_command_seq_ = 0;
    int stale_warning_count_ = 0;
    bool maintain_hover_hold_ = false;
    Eigen::VectorXd paused_hover_state_ = Eigen::VectorXd::Zero(13);
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
