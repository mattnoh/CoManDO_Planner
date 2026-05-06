#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#define private public
#include "planner_core/planner_core.hpp"
#undef private

namespace {

using Clock = std::chrono::steady_clock;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Eigen::VectorXd state(double x, double vx = 0.0) {
    Eigen::VectorXd s = Eigen::VectorXd::Zero(13);
    s(0) = x;
    s(3) = vx;
    s(6) = 1.0;
    return s;
}

Eigen::VectorXd control(double thrust) {
    Eigen::VectorXd u = Eigen::VectorXd::Zero(4);
    u(0) = thrust;
    return u;
}

SolverResult solved(const Clock::time_point& finish, double x_offset) {
    SolverResult r;
    r.success = true;
    r.solve_finish_time = finish;
    r.solve_start_time = finish - std::chrono::milliseconds(20);
    r.solve_timestamp = r.solve_start_time;
    r.solve_time_ms = 20.0;
    r.solve_iters = 2;
    r.constraint_error = 0.0;
    r.state_trajectory = {state(x_offset), state(x_offset + 1.0), state(x_offset + 2.0)};
    r.control_trajectory = {control(9.81 + x_offset), control(10.81 + x_offset)};
    return r;
}

planner_core::PlannerCoreConfig hoverConfig() {
    planner_core::PlannerCoreConfig cfg;
    cfg.ocp_type = "hover";
    cfg.platform = planner_core::Platform::Crazyflie;
    cfg.solver_type = "alipddp";
    cfg.ocp_dt = 0.1;
    cfg.n_replay = 2;
    cfg.skip_trajectory_validation = false;
    cfg.max_constraint_error = 1.0;
    cfg.enable_terminal_freeze = false;
    cfg.terminal_position_abs = Eigen::Vector3d::Zero();
    cfg.solve_lead_guard_sec = 0.0;
    cfg.initial_solve_lead_sec = 0.02;
    return cfg;
}

planner_core::PlannerCoreInput inputAt(const Clock::time_point& now) {
    planner_core::PlannerCoreInput input;
    input.current_state = state(0.0);
    input.active_odom_mode = OCPDescriptor::DroneOdomMode::Absolute;
    input.now = now;
    input.ros_time_sec = 1.0;
    input.target_snapshot.valid = true;
    return input;
}

}  // namespace

int main() {
    try {
        const auto t0 = Clock::now();

        {
            planner_core::PlannerCore core;
            auto cfg = hoverConfig();
            cfg.ocp_type = "tracking_bodyrate_tf_imu";
            cfg.skip_trajectory_validation = true;
            core.configure(cfg);
            auto input = inputAt(t0);
            input.active_odom_mode = OCPDescriptor::DroneOdomMode::BodyFrameRelative;
            input.active_odom_topic = "/drone/body_relative_odom";
            const auto out = core.trySolve(input);
            require(!out.solve_accepted, "frame mismatch should reject solve");
            require(out.rejection_reason == "frame_contract", "frame mismatch reason");
        }

        {
            planner_core::PlannerCore core;
            auto cfg = hoverConfig();
            cfg.ocp_type = "tracking_bodyrate_tf_imu";
            cfg.skip_trajectory_validation = true;
            cfg.max_first_solve_age_sec = 1.0;
            core.configure(cfg);
            auto input = inputAt(t0);
            input.active_odom_mode = OCPDescriptor::DroneOdomMode::TargetFrameRelative;
            input.active_odom_topic = "/drone/target_frame_odom";
            auto stale = solved(t0, 0.0);
            stale.solve_time_ms = 2000.0;
            auto out = core.acceptSolvedResult(input, input.current_state, stale);
            require(!out.solve_accepted, "stale first relative solve should reject");
            require(out.rejection_reason == "first_solve_stale",
                    "stale first relative solve rejection reason");
        }

        {
            planner_core::PlannerCore core;
            auto cfg = hoverConfig();
            cfg.ocp_type = "tracking_bodyrate_tf_imu";
            cfg.skip_trajectory_validation = true;
            core.configure(cfg);
            auto input = inputAt(t0);
            input.active_odom_mode = OCPDescriptor::DroneOdomMode::TargetFrameRelative;
            input.active_odom_topic = "/drone/target_frame_odom";
            auto bad = solved(t0, 0.0);
            bad.state_trajectory.back()(2) = -20.0;
            auto out = core.acceptSolvedResult(input, input.current_state, bad);
            require(!out.solve_accepted, "divergent relative trajectory should reject");
            require(out.rejection_reason == "relative_position_diverged" ||
                        out.rejection_reason == "relative_vertical_diverged",
                    "divergent relative trajectory rejection reason");
        }

        {
            planner_core::PlannerCore core;
            core.configure(hoverConfig());
            auto input = inputAt(t0);
            auto out = core.acceptSolvedResult(input, input.current_state, solved(t0, 0.0));
            require(out.solve_accepted, "first plan should be accepted");
            auto replay0 = core.sampleReplay(t0, input.target_snapshot);
            require(replay0.has_replay_sample, "first plan should activate immediately");
            require(std::abs(replay0.command.state(0)) < 1e-9, "first plan should sample node 0");
        }

        {
            planner_core::PlannerCore core;
            core.configure(hoverConfig());
            auto input = inputAt(t0);
            core.acceptSolvedResult(input, input.current_state, solved(t0, 0.0));
            core.acceptSolvedResult(input, input.current_state, solved(t0 + std::chrono::milliseconds(50), 10.0));
            auto before = core.sampleReplay(t0 + std::chrono::milliseconds(100), input.target_snapshot);
            require(before.has_replay_sample, "pending swap before sample should have plan");
            require(before.command.state(0) < 10.0, "pending plan activated too early");
            auto at_handoff = core.sampleReplay(t0 + std::chrono::milliseconds(200), input.target_snapshot);
            require(at_handoff.has_replay_sample, "handoff sample should have plan");
            require(at_handoff.command.state(0) >= 10.0, "pending plan did not activate at handoff");
            bool saw_handoff = false;
            for (const auto& d : at_handoff.diagnostics) {
                saw_handoff = saw_handoff || d.code == "handoff";
            }
            require(saw_handoff, "handoff diagnostic missing");
        }

        {
            planner_core::PlannerCore core;
            core.configure(hoverConfig());
            auto input = inputAt(t0);
            core.acceptSolvedResult(input, input.current_state, solved(t0, 0.0));
            auto bad = solved(t0 + std::chrono::milliseconds(50), 5.0);
            bad.constraint_error = 100.0;
            auto rejected = core.acceptSolvedResult(input, input.current_state, bad);
            require(!rejected.solve_accepted, "bad solve should reject");
            auto replay = core.sampleReplay(t0 + std::chrono::milliseconds(100), input.target_snapshot);
            require(replay.has_replay_sample, "previous plan should remain after rejection");
            require(replay.command.state(0) < 5.0, "rejected plan replaced active plan");
        }

        {
            planner_core::PlannerCore core;
            auto cfg = hoverConfig();
            cfg.recovery_pos_err = 1000.0;
            cfg.recovery_vel_err = 1000.0;
            core.configure(cfg);
            auto input = inputAt(t0);
            core.acceptSolvedResult(input, input.current_state, solved(t0, 0.0));

            auto early = inputAt(t0 + std::chrono::milliseconds(150));
            planner_core::PlannerCoreStepResult early_out;
            planner_core::PlannerCore::HandoffSchedule early_handoff;
            require(!core.prepareHandoffSchedule(early, early_out, early_handoff),
                    "solve should wait until lead window");
            require(early_out.rejection_reason == "waiting_for_handoff_time",
                    "wrong early solve rejection reason");

            auto handoff = inputAt(t0 + std::chrono::milliseconds(180));
            planner_core::PlannerCoreStepResult out;
            planner_core::PlannerCore::HandoffSchedule schedule;
            require(core.prepareHandoffSchedule(handoff, out, schedule),
                    "solve should start in lead window");
            double hp = 0.0, hv = 0.0;
            Eigen::VectorXd x0 = core.selectInitialState(
                handoff.current_state, handoff.target_snapshot, schedule,
                &hp, &hv, out.x0_source, out);
            require(out.x0_source == "predicted_handoff",
                    "lead-window solve should use predicted handoff state");
            require(x0.size() > 0 && std::abs(x0(0) - 2.0) < 1e-6,
                    "predicted handoff did not sample active plan at activation elapsed");
        }

        {
            planner_core::PlannerCore core;
            auto cfg = hoverConfig();
            cfg.recovery_pos_err = 0.25;
            cfg.recovery_vel_err = 1.0;
            core.configure(cfg);
            auto input = inputAt(t0);
            core.acceptSolvedResult(input, input.current_state, solved(t0, 0.0));

            auto recovery = inputAt(t0 + std::chrono::milliseconds(180));
            recovery.current_state = state(100.0);
            planner_core::PlannerCoreStepResult out;
            planner_core::PlannerCore::HandoffSchedule schedule;
            require(core.prepareHandoffSchedule(recovery, out, schedule),
                    "recovery should be inside lead window");
            double hp = 0.0, hv = 0.0;
            Eigen::VectorXd x0 = core.selectInitialState(
                recovery.current_state, recovery.target_snapshot, schedule,
                &hp, &hv, out.x0_source, out);
            require(out.x0_source == "recovery_live",
                    "large tracking error should use explicit recovery_live source");
            require(x0.size() > 0 && std::abs(x0(0) - 100.0) < 1e-6,
                    "recovery_live did not use live state");
        }

        {
            planner_core::PlannerCore core;
            auto cfg = hoverConfig();
            cfg.recovery_pos_err = 0.05;
            cfg.recovery_vel_err = 1.0;
            core.configure(cfg);
            auto input = inputAt(t0);
            core.acceptSolvedResult(input, input.current_state, solved(t0, 0.0));

            auto handoff = inputAt(t0 + std::chrono::milliseconds(180));
            handoff.current_state = state(1.8);
            planner_core::PlannerCoreStepResult out;
            planner_core::PlannerCore::HandoffSchedule schedule;
            require(core.prepareHandoffSchedule(handoff, out, schedule),
                    "handoff should be inside lead window");
            double hp = 0.0, hv = 0.0;
            Eigen::VectorXd x0 = core.selectInitialState(
                handoff.current_state, handoff.target_snapshot, schedule,
                &hp, &hv, out.x0_source, out);
            require(out.x0_source == "predicted_handoff",
                    "tracking error must compare against active plan at now, not future handoff");
            require(hp < 1e-6, "tracking error should be near zero at current active-plan time");
            require(x0.size() > 0 && std::abs(x0(0) - 2.0) < 1e-6,
                    "selected x0 should still be the future handoff state");
        }

        {
            OCPDescriptor desc;
            desc.drone_odom_mode = OCPDescriptor::DroneOdomMode::TargetFrameRelative;
            TargetSnapshot target;
            target.valid = true;
            target.position = Eigen::Vector3d(1.0, 2.0, 2.0);
            target.velocity = Eigen::Vector3d(0.3, 0.0, 0.0);
            target.angular_velocity = Eigen::Vector3d(0.0, 0.0, 1.0);
            target.orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
            Eigen::VectorXd x = state(0.0);
            x.segment(0, 3) = Eigen::Vector3d(0.0, 1.0, 0.2);
            x.segment(3, 3) = Eigen::Vector3d(0.2, 0.0, 0.0);
            const Eigen::Vector4d hover =
                planner_core::makeCrazyflieHoverCommand(desc, x, control(9.81), target);
            require(std::abs(hover(0) + 0.5) < 1e-6,
                    "TF hover vx must include target velocity and frame-rotation term");
            require(std::abs(hover(2) - 2.2) < 1e-6,
                    "TF hover z must reconstruct world z from target-frame position");
        }

        {
            OCPDescriptor desc;
            desc.drone_odom_mode = OCPDescriptor::DroneOdomMode::BodyFrameRelative;
            TargetSnapshot target;
            target.valid = true;
            target.position = Eigen::Vector3d(1.0, 2.0, 3.0);
            target.velocity = Eigen::Vector3d(0.3, 0.0, 0.0);
            target.orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
            Eigen::VectorXd x = state(0.0);
            x.segment(0, 3) = Eigen::Vector3d(0.0, 0.0, 1.0);
            x.segment(3, 3) = Eigen::Vector3d(0.2, 0.0, 0.0);
            const Eigen::Vector4d hover =
                planner_core::makeCrazyflieHoverCommand(desc, x, control(9.81), target);
            require(std::abs(hover(0) - 0.5) < 1e-6,
                    "BF hover vx must include target velocity in drone body frame");
            require(std::abs(hover(2) - 2.0) < 1e-6,
                    "BF hover z must reconstruct drone world altitude from p_T^B");
        }
    } catch (const std::exception& e) {
        std::cerr << "PlannerCore tests failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "PlannerCore tests passed\n";
    return 0;
}
