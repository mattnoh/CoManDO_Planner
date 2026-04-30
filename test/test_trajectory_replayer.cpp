#include "trajectory_replayer.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<Eigen::VectorXd> makeFixedPlan(double offset, int nodes = 4) {
    std::vector<Eigen::VectorXd> xs;
    xs.reserve(nodes);
    for (int k = 0; k < nodes; ++k) {
        Eigen::VectorXd x(3);
        x << offset + static_cast<double>(k), 0.0, 0.0;
        xs.push_back(x);
    }
    return xs;
}

std::vector<Eigen::VectorXd> makeVariablePlan() {
    std::vector<Eigen::VectorXd> xs;
    const double times[] = {0.0, 0.05, 0.20};
    for (int k = 0; k < 3; ++k) {
        Eigen::VectorXd x(4);
        x << static_cast<double>(k), 0.0, 0.0, times[k];
        xs.push_back(x);
    }
    return xs;
}

std::vector<Eigen::VectorXd> makeControls(double offset, int count = 3) {
    std::vector<Eigen::VectorXd> us;
    us.reserve(count);
    for (int k = 0; k < count; ++k) {
        Eigen::VectorXd u(4);
        u << offset + static_cast<double>(k), 0.0, 0.0, 0.0;
        us.push_back(u);
    }
    return us;
}

void testFirstPlanActivationStartsAtNodeZero() {
    TrajectoryReplayer replayer;
    const auto base = Clock::now();
    const auto finish = base + std::chrono::milliseconds(300);
    replayer.updatePlan(makeFixedPlan(0.0), makeControls(0.0), 300.0, 0,
                        finish, finish, 0.1, 2, false, 3);

    const auto sample = replayer.sample(finish + std::chrono::milliseconds(10), 0.1);
    require(sample.has_plan, "first activation: expected active plan");
    require(sample.active_solve_num == 0, "first activation: wrong solve number");
    require(std::abs(sample.x_cmd(0) - 0.1) < 1e-6,
            "first activation: sampled too far into the horizon");
}

void testPendingPlanWaitsForActivation() {
    TrajectoryReplayer replayer;
    const auto base = Clock::now();
    replayer.updatePlan(makeFixedPlan(0.0), makeControls(0.0), 10.0, 0,
                        base, base, 0.1, 2, false, 3);
    replayer.updatePlan(makeFixedPlan(100.0), makeControls(10.0), 10.0, 1,
                        base + std::chrono::milliseconds(200),
                        base + std::chrono::milliseconds(200),
                        0.1, 2, false, 3);

    auto sample = replayer.sample(base + std::chrono::milliseconds(150), 0.1);
    require(sample.active_solve_num == 0, "pending activation: swapped too early");

    sample = replayer.sample(base + std::chrono::milliseconds(210), 0.1);
    require(sample.active_solve_num == 1, "pending activation: did not swap at handoff");
    require(sample.x_cmd(0) > 100.0 && sample.x_cmd(0) < 100.2,
            "pending activation: new plan not sampled from activation time");
}

void testVariableDtUsesStateDtSlot() {
    TrajectoryReplayer replayer;
    const auto base = Clock::now();
    replayer.updatePlan(makeVariablePlan(), makeControls(0.0, 2), 10.0, 0,
                        base, base, 0.1, 1, true, 3);

    const auto sample = replayer.sample(base + std::chrono::milliseconds(125), 0.1);
    require(std::abs(sample.x_cmd(0) - 1.5) < 1e-6,
            "variable DT: did not interpolate using x[state_dim]");
}

void testJumpDiagnostics() {
    TrajectoryReplayer replayer;
    const auto base = Clock::now();
    replayer.updatePlan(makeFixedPlan(0.0), makeControls(0.0), 10.0, 0,
                        base, base, 0.1, 1, false, 3);
    replayer.updatePlan(makeFixedPlan(20.0), makeControls(50.0), 40.0, 1,
                        base + std::chrono::milliseconds(100),
                        base + std::chrono::milliseconds(100),
                        0.1, 1, false, 3);

    (void)replayer.sample(base + std::chrono::milliseconds(110), 0.1);
    const auto diag = replayer.consumeLastHandoffDiagnostic();
    require(diag.has_value(), "jump diagnostics: no diagnostic emitted");
    require(diag->previous_solve_num == 0 && diag->new_solve_num == 1,
            "jump diagnostics: wrong solve transition");
    require(diag->state_jump_norm > 10.0, "jump diagnostics: state jump not measured");
    require(diag->control_jump_norm > 10.0, "jump diagnostics: control jump not measured");
}

}  // namespace

int main() {
    try {
        testFirstPlanActivationStartsAtNodeZero();
        testPendingPlanWaitsForActivation();
        testVariableDtUsesStateDtSlot();
        testJumpDiagnostics();
    } catch (const std::exception& e) {
        std::cerr << "TrajectoryReplayer test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "TrajectoryReplayer tests passed\n";
    return 0;
}
