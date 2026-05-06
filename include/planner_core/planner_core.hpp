/// @file planner_core/planner_core.hpp
/// @brief ROS-free MPC solve/replay orchestration.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "planner_core/ocp_registry.hpp"
#include "planner_core/types.hpp"
#include "planner_core/quadrotor_mpc.hpp"
#include "dynamics/quad_6dof_target_frame.h"
#include "ocp/ocp_tracking_circle_target.hpp"
#include "planner_core/adapters.hpp"
#include "planner_core/types.hpp"
#include "trajectory_replayer.hpp"

namespace planner_core {

class PlannerCore {
public:
    void configure(const PlannerCoreConfig& config) {
        config_ = config;
        desc_ = OCPRegistry::getDescriptor(config_.ocp_type);
        if (config_.ocp_dt <= 0.0) {
            config_.ocp_dt = desc_.dt;
        }
        if (config_.n_replay < 1) {
            config_.n_replay = desc_.default_n_replay;
        }
        if (config_.terminal_state.size() == 0) {
            setTerminalTarget(config_.terminal_position_abs);
        }
        rebuildSolver();
        configured_ = true;
        reset();
    }

    void reset() {
        replayer_.clear();
        diagnostics_.clear();
        primed_ = false;
        terminal_freeze_ = false;
        solve_count_ = 0;
        last_replan_delay_sec_ = 0.0;
        last_accepted_solve_timestamp_ = Clock::time_point{};
        solve_time_ema_sec_ = 0.0;
    }

    void setTerminalTarget(const Eigen::Vector3d& xyz) {
        config_.terminal_position_abs = xyz;
        Eigen::VectorXd terminal = Eigen::VectorXd::Zero(std::max(1, desc_.state_dim));
        if (terminal.size() >= 7) {
            terminal(0) = xyz.x();
            terminal(1) = xyz.y();
            terminal(2) = xyz.z();
            terminal(6) = 1.0;
        }
        config_.terminal_state = terminal;
        if (mpc_) {
            mpc_->setTerminalState(config_.terminal_state);
        }
    }

    PlannerCoreStepResult trySolve(const PlannerCoreInput& input) {
        PlannerCoreStepResult out;
        if (!configured_) {
            reject(out, "not_configured", "PlannerCore is not configured");
            return out;
        }
        if (config_.solver_type != "alipddp" || !mpc_) {
            reject(out, "solver_unavailable", "ALIPDDP solver is unavailable");
            return out;
        }
        const auto frame_check = validateFrameContract(
            desc_, input.active_odom_mode, input.active_odom_topic,
            config_.body_relative_odom_topic, config_.target_frame_odom_topic);
        if (!frame_check.ok) {
            reject(out, "frame_contract", frame_check.reason);
            return out;
        }

        Eigen::VectorXd x0 = prepareOcpState(desc_, input.current_state, input.target_snapshot);
        HandoffSchedule handoff;
        if (!prepareHandoffSchedule(input, out, handoff)) {
            return out;
        }
        x0 = selectInitialState(x0, input.target_snapshot, handoff,
                                &out.handoff_pos_err, &out.handoff_vel_err,
                                out.x0_source, out);

        const bool apply_registry_transform = (desc_.transform_state != nullptr);
        const double pos_err = apply_registry_transform && x0.size() >= 3
            ? x0.segment(0, 3).norm()
            : absolutePositionError(input.current_state);
        const double vel_err = apply_registry_transform && x0.size() >= 6
            ? x0.segment(3, 3).norm()
            : absoluteVelocityError(input.current_state);

        if (terminal_freeze_) {
            if (pos_err > config_.terminal_freeze_exit_pos) {
                terminal_freeze_ = false;
                out.terminal_freeze_released = true;
                addDiagnostic(out, DiagnosticSeverity::Info, "terminal_freeze_released",
                              "Terminal freeze released", {{"pos_err", pos_err}});
            } else {
                reject(out, "terminal_freeze_active", "Terminal freeze is active");
                return out;
            }
        }

        const bool require_freeze_velocity =
            config_.terminal_freeze_require_vel || (desc_.transform_state != nullptr);
        const bool freeze_condition = (pos_err < config_.terminal_freeze_enter_pos) &&
            (!require_freeze_velocity || vel_err < config_.terminal_freeze_enter_vel);
        if (config_.enable_terminal_freeze && primed_ && freeze_condition) {
            terminal_freeze_ = true;
            out.terminal_freeze_engaged = true;
            reject(out, "terminal_freeze_engaged", "Terminal freeze engaged");
            addDiagnostic(out, DiagnosticSeverity::Info, "terminal_freeze_engaged",
                          "Terminal freeze engaged",
                          {{"pos_err", pos_err},
                           {"vel_err", vel_err},
                           {"require_vel", require_freeze_velocity ? 1.0 : 0.0}});
            return out;
        }

        PlannerConfig planner_config;
        planner_config.ocp_type = config_.ocp_type;
        planner_config.solver_type = config_.solver_type;
        planner_config.mode = "mpc";
        planner_config.n_replay = config_.n_replay;
        planner_config.ocp_dt = config_.ocp_dt;
        planner_config.dt = config_.ocp_dt;
        planner_config.mass_kg = desc_.default_mass_kg;
        planner_config.t_start_abs = input.ros_time_sec;
        if (input.has_target_accel_buffer) {
            planner_config.target_accel_buffer = input.target_accel_buffer;
        }

        std::any extra;
        if (desc_.prepare_extra) {
            extra = desc_.prepare_extra(planner_config, input.ros_time_sec, input.target_snapshot);
        }

        const Eigen::Vector3d target_accel =
            input.target_snapshot.valid ? input.target_snapshot.acceleration : Eigen::Vector3d::Zero();
        const auto warm_start_before_solve = mpc_->snapshotWarmStart();
        auto raw = mpc_->solve(x0, target_accel, extra, input.ros_time_sec);
        SolverResult result = toSolverResult(raw, input.target_snapshot);
        if (desc_.post_process_result) {
            desc_.post_process_result(result, input.target_snapshot);
        }
        const bool keep_controls_seed = shouldKeepRejectedControlsSeed(result);
        PlannerCoreStepResult solved_out = acceptSolvedResult(input, x0, result, out);
        if (!solved_out.solve_accepted) {
            if (keep_controls_seed && solved_out.rejection_reason == "first_solve_stale") {
                mpc_->keepLatestAsUnexecutedWarmStart();
            } else {
                mpc_->restoreWarmStart(warm_start_before_solve);
            }
        }
        return solved_out;
    }

    PlannerCoreStepResult acceptSolvedResult(const PlannerCoreInput& input,
                                             const Eigen::VectorXd& x0,
                                             SolverResult result,
                                             PlannerCoreStepResult out = {}) {
        (void)input;
        out.solve_attempted = true;
        out.solve_num = solve_count_;
        const bool first_plan = !primed_;
        if (!isSolveAcceptable(result, out.rejection_reason, first_plan)) {
            const auto rel_stats = relativeTrajectoryStats(result.state_trajectory);
            addDiagnostic(out, DiagnosticSeverity::Warn, "solve_rejected", out.rejection_reason,
                          {{"constraint_error", result.constraint_error},
                           {"solve_time_ms", result.solve_time_ms},
                           {"active_elapsed_now", out.active_elapsed_now_sec},
                           {"activation_elapsed", out.activation_elapsed_sec},
                           {"activation_wall_time_sec", out.activation_wall_time_sec},
                           {"solve_lead_sec", out.solve_lead_sec},
                           {"max_relative_position_norm", rel_stats.max_position_norm},
                           {"max_relative_position_node", static_cast<double>(rel_stats.max_position_node)},
                           {"max_relative_vertical_abs", rel_stats.max_vertical_abs},
                           {"max_relative_vertical_node", static_cast<double>(rel_stats.max_vertical_node)},
                           {"max_relative_velocity_norm", rel_stats.max_velocity_norm},
                           {"max_relative_velocity_node", static_cast<double>(rel_stats.max_velocity_node)},
                           {"relative_position_limit", config_.max_relative_position_norm},
                           {"relative_vertical_limit", config_.max_relative_vertical_abs},
                           {"relative_velocity_limit", config_.max_relative_velocity_norm}});
            out.solve_result = std::move(result);
            return out;
        }

        const auto previous_plan_origin = last_accepted_solve_timestamp_;
        const double previous_replan_delay_sec = last_replan_delay_sec_;
        last_replan_delay_sec_ = replayAdvanceTime(result.state_trajectory);

        Clock::time_point plan_origin_time = result.solve_finish_time;
        if (!first_plan) {
            if (out.activation_elapsed_sec > 0.0 && previous_plan_origin != Clock::time_point{}) {
                plan_origin_time = previous_plan_origin + std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(out.activation_elapsed_sec));
            } else {
                plan_origin_time = previous_plan_origin + std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(previous_replan_delay_sec));
            }
        }
        const auto earliest_activation_time = plan_origin_time;
        out.solve_finish_late_by_sec =
            std::chrono::duration<double>(result.solve_finish_time - plan_origin_time).count();

        replayer_.updatePlan(result.state_trajectory,
                             result.control_trajectory,
                             result.solve_time_ms,
                             solve_count_,
                             plan_origin_time,
                             earliest_activation_time,
                             config_.ocp_dt,
                             config_.n_replay,
                             desc_.variable_dt,
                             desc_.state_dim);

        last_accepted_solve_timestamp_ = plan_origin_time;
        primed_ = true;
        updateSolveTimeEma(result.solve_time_ms);
        out.solve_accepted = true;
        out.replan_delay_sec = last_replan_delay_sec_;
        out.path.states = makeTrajectoryForPublishing(result);
        out.solve_result = result;
        out.solve_log = makeSolveLogData(result, solve_count_);
        addDiagnostic(out, DiagnosticSeverity::Info, "solve_accepted", "Solve accepted",
                      {{"solve_time_ms", result.solve_time_ms},
                       {"replan_delay_sec", last_replan_delay_sec_},
                       {"active_elapsed_now", out.active_elapsed_now_sec},
                       {"activation_elapsed", out.activation_elapsed_sec},
                       {"activation_wall_time_sec", out.activation_wall_time_sec},
                       {"solve_lead_sec", out.solve_lead_sec},
                       {"solve_finish_late_by_sec", out.solve_finish_late_by_sec},
                       {"x0_px", x0.size() > 0 ? x0(0) : 0.0},
                       {"x0_py", x0.size() > 1 ? x0(1) : 0.0},
                       {"x0_pz", x0.size() > 2 ? x0(2) : 0.0}});
        ++solve_count_;
        return out;
    }

    PlannerCoreStepResult sampleReplay(const std::chrono::steady_clock::time_point& now,
                                       const TargetSnapshot& target_snapshot) {
        PlannerCoreStepResult out;
        if (!primed_) {
            return out;
        }
        auto replay = replayer_.sample(now, config_.ocp_dt);
        if (!replay.has_plan) {
            return out;
        }
        out.has_replay_sample = true;
        out.solve_num = replay.active_solve_num;
        out.command = makePlannerCommand(config_.platform, desc_, replay.x_cmd, replay.u_cmd,
                                         target_snapshot);
        out.stale = replay.elapsed > replay.horizon_end + 0.2;
        if (out.stale) {
            addDiagnostic(out, DiagnosticSeverity::Warn, "trajectory_stale",
                          "Trajectory sample is past horizon",
                          {{"elapsed", replay.elapsed}, {"horizon_end", replay.horizon_end}});
        }
        if (auto diag = replayer_.consumeLastHandoffDiagnostic()) {
            PlannerDiagnostic d;
            d.severity = DiagnosticSeverity::Info;
            d.code = "handoff";
            d.message = "Active plan swapped to pending plan";
            d.values = {
                {"previous_solve_num", static_cast<double>(diag->previous_solve_num)},
                {"new_solve_num", static_cast<double>(diag->new_solve_num)},
                {"active_plan_age_sec", diag->active_plan_age_sec},
                {"solve_latency_sec", diag->solve_latency_sec},
                {"state_jump_norm", diag->state_jump_norm},
                {"control_jump_norm", diag->control_jump_norm},
            };
            const auto old_cmd = makePlannerCommand(config_.platform, desc_, diag->old_state_cmd,
                                                    diag->old_control_cmd, target_snapshot);
            const auto new_cmd = makePlannerCommand(config_.platform, desc_, diag->new_state_cmd,
                                                    diag->new_control_cmd, target_snapshot);
            if (old_cmd.kind == CommandKind::Hover && new_cmd.kind == CommandKind::Hover) {
                d.values["hover_z_jump"] = std::abs(new_cmd.hover(2) - old_cmd.hover(2));
            }
            d.state_a = diag->old_state_cmd;
            d.state_b = diag->new_state_cmd;
            d.control_a = diag->old_control_cmd;
            d.control_b = diag->new_control_cmd;
            out.diagnostics.push_back(std::move(d));
        }
        return out;
    }

    std::vector<PlannerDiagnostic> consumeDiagnostics() {
        auto out = diagnostics_;
        diagnostics_.clear();
        return out;
    }

    bool isPrimed() const { return primed_; }
    int solveCount() const { return solve_count_; }
    double lastReplanDelaySec() const { return last_replan_delay_sec_; }

private:
    using Clock = std::chrono::steady_clock;

    struct HandoffSchedule {
        bool first_plan = true;
        double active_elapsed_now = 0.0;
        double activation_elapsed = 0.0;
        double solve_lead_sec = 0.0;
        Clock::time_point active_origin{};
        Clock::time_point activation_time{};
    };

    struct RelativeTrajectoryStats {
        double max_position_norm = 0.0;
        double max_vertical_abs = 0.0;
        double max_velocity_norm = 0.0;
        int max_position_node = -1;
        int max_vertical_node = -1;
        int max_velocity_node = -1;
    };

    void rebuildSolver() {
        if (config_.solver_type != "alipddp") {
            mpc_.reset();
            return;
        }
        QuadrotorMPC::Config cfg;
        cfg.ocp_type = config_.ocp_type;
        cfg.terminal_state = config_.terminal_state;
        cfg.n_shift = config_.n_replay;
        mpc_ = std::make_unique<QuadrotorMPC>(cfg);
    }

    static SolverResult toSolverResult(const QuadrotorMPC::Result& r,
                                       const TargetSnapshot& target_snapshot) {
        SolverResult result;
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
        return result;
    }

    bool prepareHandoffSchedule(const PlannerCoreInput& input,
                                PlannerCoreStepResult& out,
                                HandoffSchedule& handoff) const {
        handoff.first_plan = !primed_;
        if (!primed_) {
            out.x0_source = "live_first";
            return true;
        }

        if (replayer_.hasPendingPlan()) {
            reject(out, "waiting_for_pending_handoff", "Pending plan has not activated yet");
            return false;
        }

        handoff.active_origin = replayer_.activePlanOriginTime();
        handoff.active_elapsed_now = replayer_.activeElapsedAt(input.now);
        handoff.activation_elapsed = last_replan_delay_sec_;
        handoff.solve_lead_sec = solveLeadSec();
        handoff.activation_time =
            handoff.active_origin + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(handoff.activation_elapsed));

        out.active_elapsed_now_sec = handoff.active_elapsed_now;
        out.activation_elapsed_sec = handoff.activation_elapsed;
        out.activation_wall_time_sec =
            std::chrono::duration<double>(handoff.activation_time.time_since_epoch()).count();
        out.solve_lead_sec = handoff.solve_lead_sec;

        const double solve_start_elapsed =
            std::max(0.0, handoff.activation_elapsed - handoff.solve_lead_sec);
        if (handoff.active_elapsed_now + 1e-6 < solve_start_elapsed) {
            reject(out, "waiting_for_handoff_time", "Active plan has not reached solve lead window");
            if (!out.diagnostics.empty()) {
                out.diagnostics.back().values = {
                    {"active_elapsed_now", handoff.active_elapsed_now},
                    {"activation_elapsed", handoff.activation_elapsed},
                    {"activation_wall_time_sec", out.activation_wall_time_sec},
                    {"solve_lead_sec", handoff.solve_lead_sec},
                    {"solve_start_elapsed", solve_start_elapsed},
                };
            }
            return false;
        }
        return true;
    }

    Eigen::VectorXd selectInitialState(const Eigen::VectorXd& live_x0,
                                       const TargetSnapshot& target_snapshot,
                                       const HandoffSchedule& handoff,
                                       double* handoff_pos_err,
                                       double* handoff_vel_err,
                                       std::string& x0_source,
                                       PlannerCoreStepResult& out) const {
        if (handoff_pos_err) *handoff_pos_err = 0.0;
        if (handoff_vel_err) *handoff_vel_err = 0.0;
        x0_source = handoff.first_plan ? "live_first" : "live";
        if (handoff.first_plan || !desc_.use_predicted_handoff_state) {
            return live_x0;
        }
        TrajectoryReplayer::ReplaySample predicted;
        if (!replayer_.sampleActiveAtElapsed(handoff.activation_elapsed, config_.ocp_dt, &predicted) ||
            !predicted.has_plan ||
            predicted.x_cmd.size() < live_x0.size()) {
            x0_source = "live_no_prediction";
            addDiagnostic(out, DiagnosticSeverity::Warn, "handoff_prediction_missing",
                          "Predicted handoff sample unavailable; using live state",
                          {{"activation_elapsed", handoff.activation_elapsed}});
            return live_x0;
        }
        const double dt_to_activation =
            std::max(0.0, handoff.activation_elapsed - handoff.active_elapsed_now);
        const TargetSnapshot predicted_target = predictTargetSnapshot(target_snapshot, dt_to_activation);
        Eigen::VectorXd predicted_x0 = preparePredictedHandoffState(predicted.x_cmd,
                                                                    predicted_target,
                                                                    live_x0.size());
        Eigen::VectorXd tracking_x0 = predicted_x0;
        TrajectoryReplayer::ReplaySample active_now;
        if (replayer_.sampleActiveAtElapsed(handoff.active_elapsed_now, config_.ocp_dt, &active_now) &&
            active_now.has_plan &&
            active_now.x_cmd.size() >= live_x0.size()) {
            tracking_x0 = preparePredictedHandoffState(active_now.x_cmd, target_snapshot, live_x0.size());
        } else {
            addDiagnostic(out, DiagnosticSeverity::Warn, "handoff_tracking_sample_missing",
                          "Active-plan sample unavailable for tracking-error check",
                          {{"active_elapsed_now", handoff.active_elapsed_now}});
        }
        double pos_err = 0.0;
        double vel_err = 0.0;
        if (live_x0.size() >= 6 && tracking_x0.size() >= 6) {
            pos_err = (live_x0.segment(0, 3) - tracking_x0.segment(0, 3)).norm();
            vel_err = (live_x0.segment(3, 3) - tracking_x0.segment(3, 3)).norm();
        }
        if (handoff_pos_err) *handoff_pos_err = pos_err;
        if (handoff_vel_err) *handoff_vel_err = vel_err;
        if (pos_err > config_.recovery_pos_err || vel_err > config_.recovery_vel_err) {
            x0_source = "recovery_live";
            addDiagnostic(out, DiagnosticSeverity::Warn, "handoff_recovery_live",
                          "Tracking error exceeds recovery threshold; replanning from live state",
                          {{"handoff_pos_err_now", pos_err},
                           {"handoff_vel_err_now", vel_err},
                           {"activation_elapsed", handoff.activation_elapsed}});
            return live_x0;
        }
        x0_source = "predicted_handoff";
        return predicted_x0;
    }

    Eigen::VectorXd preparePredictedHandoffState(const Eigen::VectorXd& predicted_sample,
                                                 const TargetSnapshot& target,
                                                 int live_size) const {
        const int out_size = std::min(live_size, static_cast<int>(predicted_sample.size()));
        Eigen::VectorXd predicted_x0 = predicted_sample.head(out_size);
        if ((desc_.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative ||
             desc_.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative) &&
            desc_.transform_state && predicted_sample.size() >= 10) {
            Eigen::VectorXd base = predicted_sample.head(10);
            Eigen::VectorXd refreshed = desc_.transform_state(base, target);
            if (refreshed.size() >= live_size) {
                predicted_x0 = refreshed.head(live_size);
            } else if (refreshed.size() > 0) {
                predicted_x0.head(std::min<int>(predicted_x0.size(), refreshed.size())) =
                    refreshed.head(std::min<int>(predicted_x0.size(), refreshed.size()));
            }
        }
        return predicted_x0;
    }

    static TargetSnapshot predictTargetSnapshot(const TargetSnapshot& target, double dt) {
        TargetSnapshot predicted = target;
        if (!target.valid || dt <= 0.0) {
            return predicted;
        }
        predicted.position = target.position + target.velocity * dt +
            0.5 * target.acceleration * dt * dt;
        predicted.velocity = target.velocity + target.acceleration * dt;
        predicted.angular_velocity = target.angular_velocity + target.angular_acceleration * dt;
        predicted.orientation = integrateQuaternion(target.orientation,
                                                    target.angular_velocity,
                                                    target.angular_acceleration,
                                                    dt);
        return predicted;
    }

    static Eigen::Vector4d integrateQuaternion(const Eigen::Vector4d& q_vec,
                                               const Eigen::Vector3d& omega0,
                                               const Eigen::Vector3d& beta,
                                               double dt) {
        Eigen::Quaterniond q(q_vec(0), q_vec(1), q_vec(2), q_vec(3));
        q.normalize();
        const Eigen::Vector3d rotvec = omega0 * dt + 0.5 * beta * dt * dt;
        const double angle = rotvec.norm();
        if (angle > 1e-12) {
            const Eigen::AngleAxisd aa(angle, rotvec / angle);
            q = q * Eigen::Quaterniond(aa);
            q.normalize();
        }
        return Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
    }

    double solveLeadSec() const {
        const double learned = (solve_time_ema_sec_ > 0.0)
            ? solve_time_ema_sec_
            : config_.initial_solve_lead_sec;
        return std::max(0.0, learned + config_.solve_lead_guard_sec);
    }

    void updateSolveTimeEma(double solve_time_ms) {
        const double solve_sec = std::max(0.0, solve_time_ms * 0.001);
        if (solve_time_ema_sec_ <= 0.0) {
            solve_time_ema_sec_ = solve_sec;
            return;
        }
        constexpr double alpha = 0.30;
        solve_time_ema_sec_ = alpha * solve_sec + (1.0 - alpha) * solve_time_ema_sec_;
    }

    double absolutePositionError(const Eigen::VectorXd& x) const {
        if (x.size() < 3) {
            return 0.0;
        }
        return (x.segment(0, 3) - config_.terminal_position_abs).norm();
    }

    static double absoluteVelocityError(const Eigen::VectorXd& x) {
        if (x.size() < 6) {
            return 0.0;
        }
        return x.segment(3, 3).norm();
    }

    double stateNodeTime(const Eigen::VectorXd& s, int index) const {
        if (s.size() > desc_.state_dim) {
            return s(desc_.state_dim);
        }
        return index * config_.ocp_dt;
    }

    double replayAdvanceTime(const std::vector<Eigen::VectorXd>& traj) const {
        if (traj.size() < 2) {
            return std::max(config_.ocp_dt, 1e-3);
        }
        const int idx = std::min(config_.n_replay, static_cast<int>(traj.size()) - 1);
        const double t_adv = stateNodeTime(traj[idx], idx);
        return std::max(t_adv, std::max(config_.ocp_dt, 1e-3));
    }

    bool isSolveAcceptable(const SolverResult& result,
                           std::string& reason,
                           bool first_plan) const {
        const bool relative_mode =
            desc_.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative ||
            desc_.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative;
        if (first_plan && relative_mode &&
            result.solve_time_ms * 0.001 > config_.max_first_solve_age_sec) {
            reason = "first_solve_stale";
            return false;
        }
        if (!result.success || result.state_trajectory.size() < 2) {
            reason = "solve_failed";
            return false;
        }
        if (result.constraint_error > config_.max_constraint_error) {
            reason = "constraint_error";
            return false;
        }
        if (!validateRelativeTrajectorySanity(result.state_trajectory, reason)) {
            return false;
        }
        if (config_.skip_trajectory_validation || desc_.skip_trajectory_validation) {
            return true;
        }
        if (!validateTrajectory(result.state_trajectory, result.control_trajectory)) {
            reason = "physical_bounds";
            return false;
        }
        return true;
    }

    bool shouldKeepRejectedControlsSeed(const SolverResult& result) const {
        const bool relative_mode =
            desc_.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative ||
            desc_.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative;
        if (primed_ || !relative_mode || !result.success ||
            result.solve_time_ms * 0.001 <= config_.max_first_solve_age_sec ||
            result.constraint_error > config_.max_constraint_error) {
            return false;
        }
        std::string sanity_reason;
        return validateRelativeTrajectorySanity(result.state_trajectory, sanity_reason);
    }

    bool validateRelativeTrajectorySanity(const std::vector<Eigen::VectorXd>& X,
                                          std::string& reason) const {
        const bool relative_mode =
            desc_.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative ||
            desc_.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative;
        if (!relative_mode) {
            return true;
        }
        for (const auto& x : X) {
            if (x.size() < 6) {
                continue;
            }
            for (int i = 0; i < std::min<int>(x.size(), desc_.state_dim); ++i) {
                if (!std::isfinite(x(i))) {
                    reason = "relative_state_nonfinite";
                    return false;
                }
            }
            if (x.segment(0, 3).norm() > config_.max_relative_position_norm) {
                reason = "relative_position_diverged";
                return false;
            }
            if (std::abs(x(2)) > config_.max_relative_vertical_abs) {
                reason = "relative_vertical_diverged";
                return false;
            }
            if (x.segment(3, 3).norm() > config_.max_relative_velocity_norm) {
                reason = "relative_velocity_diverged";
                return false;
            }
        }
        return true;
    }

    RelativeTrajectoryStats relativeTrajectoryStats(const std::vector<Eigen::VectorXd>& X) const {
        RelativeTrajectoryStats stats;
        for (int i = 0; i < static_cast<int>(X.size()); ++i) {
            const auto& x = X[i];
            if (x.size() < 6) {
                continue;
            }
            const double p_norm = x.segment(0, 3).norm();
            const double z_abs = std::abs(x(2));
            const double v_norm = x.segment(3, 3).norm();
            if (p_norm > stats.max_position_norm) {
                stats.max_position_norm = p_norm;
                stats.max_position_node = i;
            }
            if (z_abs > stats.max_vertical_abs) {
                stats.max_vertical_abs = z_abs;
                stats.max_vertical_node = i;
            }
            if (v_norm > stats.max_velocity_norm) {
                stats.max_velocity_norm = v_norm;
                stats.max_velocity_node = i;
            }
        }
        return stats;
    }

    bool validateTrajectory(const std::vector<Eigen::VectorXd>& X,
                            const std::vector<Eigen::VectorXd>& U) const {
        if (X.size() < 2) return false;
        for (const auto& x : X) {
            if (x.size() < desc_.state_dim) continue;
            if (!desc_.skip_altitude_validation && x.size() >= 3 && x(2) < -0.05) return false;
            if (x.size() >= 6 && x.segment(3, 3).norm() > 20.0) return false;
            if (x.size() >= 13 && x.segment(10, 3).norm() > 50.0) return false;
        }
        for (const auto& u : U) {
            if (u.size() > 0 && (u(0) < -0.1 || u(0) > 50.0)) {
                return false;
            }
        }
        return true;
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

    std::vector<Eigen::VectorXd> reconstructTrackingCircleTargetAbsoluteTrajectory(
        const std::vector<Eigen::VectorXd>& rel_traj,
        const Eigen::Vector3d& target_pos0,
        const Eigen::Vector3d& target_vel0,
        const target_models::TargetAccelBuffer& buffer,
        double t0_abs) const {
        std::vector<Eigen::VectorXd> abs_traj;
        abs_traj.reserve(rel_traj.size());
        Eigen::Vector3d tgt_pos = target_pos0;
        Eigen::Vector3d tgt_vel = target_vel0;
        double prev_t_node = 0.0;
        for (int i = 0; i < static_cast<int>(rel_traj.size()); ++i) {
            const Eigen::VectorXd& x_rel = rel_traj[i];
            if (x_rel.size() < desc_.state_dim) {
                abs_traj.push_back(x_rel);
                continue;
            }
            const double raw_t_node = stateNodeTime(x_rel, i);
            const double t_node = std::max(prev_t_node, raw_t_node);
            integrateTargetSegment(tgt_pos, tgt_vel, t0_abs + prev_t_node, t0_abs + t_node, buffer);
            prev_t_node = t_node;
            Eigen::VectorXd x_abs = x_rel;
            x_abs.segment(0, 3) = x_rel.segment(0, 3) + tgt_pos;
            x_abs.segment(3, 3) = x_rel.segment(3, 3) + tgt_vel;
            abs_traj.push_back(x_abs);
        }
        return abs_traj;
    }

    std::vector<Eigen::VectorXd> makeTrajectoryForPublishing(const SolverResult& result) const {
        if (desc_.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyFrameRelative) {
            if (!result.is_relative_plan ||
                result.target_world_pos_trajectory.size() != result.state_trajectory.size() ||
                result.target_world_vel_trajectory.size() != result.state_trajectory.size()) {
                return result.state_trajectory;
            }
            std::vector<Eigen::VectorXd> traj_abs = result.state_trajectory;
            for (int i = 0; i < static_cast<int>(traj_abs.size()); ++i) {
                auto& s = traj_abs[i];
                if (s.size() < desc_.state_dim) continue;
                const double t_node = stateNodeTime(s, i);
                const Eigen::AngleAxisd yaw_step(
                    result.target_snapshot_omega.z() * t_node, Eigen::Vector3d::UnitZ());
                const Eigen::Matrix3d R_WN =
                    Quad6DOFVarTime<double>::calcC(result.target_snapshot_quat) *
                    yaw_step.toRotationMatrix();
                const Eigen::Matrix3d R_WB =
                    R_WN * Quad6DOFVarTime<double>::calcC(s.segment(6, 4));
                s.segment(0, 3) = result.target_world_pos_trajectory[i] - R_WB * s.segment(0, 3);
                s.segment(3, 3) = result.target_world_vel_trajectory[i] + R_WB * s.segment(3, 3);
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
                if (s.size() < desc_.state_dim) continue;
                if (desc_.drone_odom_mode == OCPDescriptor::DroneOdomMode::TargetFrameRelative &&
                    result.target_snapshot_quat.size() == 4) {
                    const double t_node = stateNodeTime(s, i);
                    const Eigen::Vector4d q_wxyz = result.target_snapshot_quat;
                    const Eigen::AngleAxisd yaw_step(
                        result.target_snapshot_omega.z() * t_node, Eigen::Vector3d::UnitZ());
                    const Eigen::Matrix3d R_WN =
                        Quad6DOFVarTime<double>::calcC(q_wxyz) * yaw_step.toRotationMatrix();
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
        if (config_.ocp_type == "tracking_circle_target") {
            TrackingCircleTargetOCP::TrackingCircleTargetExtra ex;
            if (tryGetTrackingCircleTargetExtra(result.extra, ex)) {
                return reconstructTrackingCircleTargetAbsoluteTrajectory(
                    result.state_trajectory, result.target_snapshot_pos,
                    result.target_snapshot_vel, ex.buf, ex.t_abs);
            }
        }
        std::vector<Eigen::VectorXd> traj_abs = result.state_trajectory;
        for (int i = 0; i < static_cast<int>(traj_abs.size()); ++i) {
            auto& s = traj_abs[i];
            if (s.size() < desc_.state_dim) continue;
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

    PlannerSolveLogData makeSolveLogData(const SolverResult& result, int solve_num) const {
        PlannerSolveLogData meta;
        meta.solve_num = solve_num;
        meta.solve_time_ms = result.solve_time_ms;
        meta.solve_iters = result.solve_iters;
        meta.is_relative_plan = result.is_relative_plan;
        meta.ocp_dt = config_.ocp_dt;
        meta.target_snapshot_pos = result.target_snapshot_pos;
        meta.target_snapshot_vel = result.target_snapshot_vel;
        meta.target_snapshot_acc = result.target_snapshot_acc;
        meta.target_snapshot_quat = result.target_snapshot_quat;
        meta.target_snapshot_omega = result.target_snapshot_omega;
        meta.target_snapshot_beta = result.target_snapshot_beta;
        meta.target_world_pos_trajectory = result.target_world_pos_trajectory;
        meta.target_world_vel_trajectory = result.target_world_vel_trajectory;
        meta.coord_mode = frameLabel(desc_.drone_odom_mode);
        if (result.is_relative_plan) {
            meta.coord_mode = "absolute_shifted";
        }
        return meta;
    }

    void reject(PlannerCoreStepResult& out, std::string code, std::string message) const {
        out.rejection_reason = code;
        addDiagnostic(out, DiagnosticSeverity::Warn, std::move(code), std::move(message), {});
    }

    static void addDiagnostic(PlannerCoreStepResult& out,
                              DiagnosticSeverity severity,
                              std::string code,
                              std::string message,
                              std::map<std::string, double> values) {
        PlannerDiagnostic d;
        d.severity = severity;
        d.code = std::move(code);
        d.message = std::move(message);
        d.values = std::move(values);
        out.diagnostics.push_back(std::move(d));
    }

    PlannerCoreConfig config_;
    OCPDescriptor desc_;
    bool configured_ = false;
    bool primed_ = false;
    bool terminal_freeze_ = false;
    int solve_count_ = 0;
    double last_replan_delay_sec_ = 0.0;
    double solve_time_ema_sec_ = 0.0;
    Clock::time_point last_accepted_solve_timestamp_{};
    TrajectoryReplayer replayer_;
    std::unique_ptr<QuadrotorMPC> mpc_;
    std::vector<PlannerDiagnostic> diagnostics_;
};

}  // namespace planner_core
