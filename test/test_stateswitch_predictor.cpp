#include "ocp/ocp_stateswitch.hpp"

#include <Eigen/Dense>

#include <any>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool near(const Eigen::Vector3d& a, const Eigen::Vector3d& b, double tol = 1e-12) {
    return (a - b).norm() <= tol;
}

Eigen::VectorXd state13() {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(13);
    x(2) = 1.0;
    x(6) = 1.0;
    return x;
}

}  // namespace

int main() {
    try {
        StateswitchOCP::TargetPredictor predictor;
        predictor.position = Eigen::Vector3d(1.0, 2.0, 3.0);
        predictor.velocity = Eigen::Vector3d(0.1, 0.2, 0.3);
        predictor.acceleration = Eigen::Vector3d(0.4, 0.5, 0.6);

        require(near(predictor.predictAccel(0.0), predictor.acceleration),
                "predictor should expose snapshot accel at solve start");
        require(near(predictor.predictAccel(2.0), predictor.acceleration),
                "zero jerk/snap predictor should keep acceleration constant");
        require(near(predictor.predictVel(2.0),
                     predictor.velocity + 2.0 * predictor.acceleration),
                "zero jerk/snap predictor should integrate velocity");

        auto desc = StateswitchOCP::descriptor();
        PlannerConfig cfg;
        TargetSnapshot target;
        target.valid = true;
        target.position = predictor.position;
        target.velocity = predictor.velocity;
        target.acceleration = predictor.acceleration;

        auto extra_any = desc.prepare_extra(cfg, 12.0, target);
        auto extra = std::any_cast<StateswitchOCP::StateswitchExtra>(extra_any);
        require(static_cast<bool>(extra.predictor), "stateswitch extra should carry predictor");
        require(near(extra.predictor->predictAccel(1.0), predictor.acceleration),
                "stateswitch predictor should come from target snapshot");

        OCPCreateArgs args;
        args.current_state = state13();
        args.terminal_state = state13();
        args.target_accel = Eigen::Vector3d(0.7, 0.8, 0.9);
        args.extra = extra_any;
        auto ocp = desc.create(args);
        require(static_cast<bool>(ocp), "stateswitch OCP should build without TargetAccelBuffer");

        args.extra.reset();
        auto fallback_ocp = desc.create(args);
        require(static_cast<bool>(fallback_ocp), "stateswitch fallback should build without extra predictor");
    } catch (const std::exception& e) {
        std::cerr << "Stateswitch predictor tests failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Stateswitch predictor tests passed\n";
    return 0;
}
