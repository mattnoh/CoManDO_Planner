#include "planner_core/planner_core.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

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
    } catch (const std::exception& e) {
        std::cerr << "PlannerCore tests failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "PlannerCore tests passed\n";
    return 0;
}
