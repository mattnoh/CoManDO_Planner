/// @file planner_core/planner_core.hpp
/// @brief ROS-free MPC solve/replay orchestration.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <any>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/ocp_registry.hpp"
#include "core/planner_config.hpp"
#include "core/quadrotor_mpc.hpp"
#include "dynamics/quad_6dof_target_frame.h"
#include "ocp/ocp_tracking_circle_target.hpp"
#include "planner_core/command_adapter.hpp"
#include "planner_core/frame_adapter.hpp"
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
        if (primed_) {
            const double elapsed_since_solve =
                std::chrono::duration<double>(input.now - last_accepted_solve_timestamp_).count();
            if (elapsed_since_solve + 1e-6 < last_replan_delay_sec_) {
                reject(out, "waiting_for_handoff_time", "Active plan has not reached replan point");
                return out;
            }
        }

        const auto frame_check = validateFrameContract(
            desc_, input.active_odom_mode, input.active_odom_topic,
            config_.body_relative_odom_topic, config_.target_frame_odom_topic);
        if (!frame_check.ok) {
            reject(out, "frame_contract", frame_check.reason);
            return out;
        }

        Eigen::VectorXd x0 = prepareOcpState(desc_, input.current_state, input.target_snapshot);
        x0 = selectInitialState(x0, &out.handoff_pos_err, &out.handoff_vel_err, out.x0_source);

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

        const bool freeze_condition = (pos_err < config_.terminal_freeze_enter_pos) &&
            (!config_.terminal_freeze_require_vel || vel_err < config_.terminal_freeze_enter_vel);
        if (config_.enable_terminal_freeze && primed_ && freeze_condition) {
            terminal_freeze_ = true;
            out.terminal_freeze_engaged = true;
            reject(out, "terminal_freeze_engaged", "Terminal freeze engaged");
            addDiagnostic(out, DiagnosticSeverity::Info, "terminal_freeze_engaged",
                          "Terminal freeze engaged", {{"pos_err", pos_err}, {"vel_err", vel_err}});
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
        auto raw = mpc_->solve(x0, target_accel, extra, input.ros_time_sec);
        SolverResult result = toSolverResult(raw, input.target_snapshot);
        if (desc_.post_process_result) {
            desc_.post_process_result(result, input.target_snapshot);
        }
        return acceptSolvedResult(input, x0, result, out);
    }

    PlannerCoreStepResult acceptSolvedResult(const PlannerCoreInput& input,
                                             const Eigen::VectorXd& x0,
                                             SolverResult result,
                                             PlannerCoreStepResult out = {}) {
        (void)input;
        out.solve_attempted = true;
        out.solve_num = solve_count_;
        if (!isSolveAcceptable(result, out.rejection_reason)) {
            addDiagnostic(out, DiagnosticSeverity::Warn, "solve_rejected", out.rejection_reason,
                          {{"constraint_error", result.constraint_error}});
            out.solve_result = std::move(result);
            return out;
        }

        const auto previous_plan_origin = last_accepted_solve_timestamp_;
        const double previous_replan_delay_sec = last_replan_delay_sec_;
        const bool first_plan = !primed_;
        last_replan_delay_sec_ = replayAdvanceTime(result.state_trajectory);

        const auto plan_origin_time = first_plan
            ? result.solve_finish_time
            : previous_plan_origin + std::chrono::duration_cast<Clock::duration>(
                  std::chrono::duration<double>(previous_replan_delay_sec));
        const auto earliest_activation_time = plan_origin_time;

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
        out.solve_accepted = true;
        out.replan_delay_sec = last_replan_delay_sec_;
        out.path.states = makeTrajectoryForPublishing(result);
        out.solve_result = result;
        out.solve_log = makeSolveLogData(result, solve_count_);
        addDiagnostic(out, DiagnosticSeverity::Info, "solve_accepted", "Solve accepted",
                      {{"solve_time_ms", result.solve_time_ms},
                       {"replan_delay_sec", last_replan_delay_sec_},
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

    Eigen::VectorXd selectInitialState(const Eigen::VectorXd& live_x0,
                                       double* handoff_pos_err,
                                       double* handoff_vel_err,
                                       std::string& x0_source) const {
        if (handoff_pos_err) *handoff_pos_err = 0.0;
        if (handoff_vel_err) *handoff_vel_err = 0.0;
        x0_source = "live";
        if (!primed_ || !desc_.use_predicted_handoff_state) {
            return live_x0;
        }
        TrajectoryReplayer::ReplaySample predicted;
        if (!replayer_.sampleActiveAtElapsed(last_replan_delay_sec_, config_.ocp_dt, &predicted) ||
            !predicted.has_plan ||
            predicted.x_cmd.size() < live_x0.size()) {
            x0_source = "live_no_prediction";
            return live_x0;
        }
        Eigen::VectorXd predicted_x0 = predicted.x_cmd.head(live_x0.size());
        double pos_err = 0.0;
        double vel_err = 0.0;
        if (live_x0.size() >= 6 && predicted_x0.size() >= 6) {
            pos_err = (live_x0.segment(0, 3) - predicted_x0.segment(0, 3)).norm();
            vel_err = (live_x0.segment(3, 3) - predicted_x0.segment(3, 3)).norm();
        }
        if (handoff_pos_err) *handoff_pos_err = pos_err;
        if (handoff_vel_err) *handoff_vel_err = vel_err;
        if (pos_err > 0.25 || vel_err > 1.0) {
            x0_source = "live_tracking_error";
            return live_x0;
        }
        x0_source = "predicted_handoff";
        return predicted_x0;
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

    bool isSolveAcceptable(const SolverResult& result, std::string& reason) const {
        if (config_.skip_trajectory_validation || desc_.skip_trajectory_validation) {
            return true;
        }
        if (!result.success || result.state_trajectory.size() < 2) {
            reason = "solve_failed";
            return false;
        }
        if (result.constraint_error > config_.max_constraint_error) {
            reason = "constraint_error";
            return false;
        }
        if (!validateTrajectory(result.state_trajectory, result.control_trajectory)) {
            reason = "physical_bounds";
            return false;
        }
        return true;
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
                const Eigen::Matrix3d R_WB = Quad6DOFVarTime<double>::calcC(s.segment(6, 4));
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
                s.segment(0, 3) += result.target_world_pos_trajectory[i];
                s.segment(3, 3) += result.target_world_vel_trajectory[i];
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
    Clock::time_point last_accepted_solve_timestamp_{};
    TrajectoryReplayer replayer_;
    std::unique_ptr<QuadrotorMPC> mpc_;
    std::vector<PlannerDiagnostic> diagnostics_;
};

}  // namespace planner_core
