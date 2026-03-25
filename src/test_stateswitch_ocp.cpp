#include "quadrotor_mpc.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

struct CircularTarget {
    Eigen::Vector3d center{0.0, 0.0, 1.5};
    double R = 0.0;
    double omega = 0.0;
    double phi0 = 0.0;
    Eigen::Vector3d pos(double t) const {
        double ph = omega * t + phi0;
        return center + Eigen::Vector3d(R * std::cos(ph), R * std::sin(ph), 0.0);
    }
    Eigen::Vector3d vel(double t) const {
        double ph = omega * t + phi0;
        return Eigen::Vector3d(-R * omega * std::sin(ph), R * omega * std::cos(ph), 0.0);
    }
    Eigen::Vector3d accel(double t) const {
        double ph = omega * t + phi0;
        return Eigen::Vector3d(-R * omega * omega * std::cos(ph),
                               -R * omega * omega * std::sin(ph), 0.0);
    }
};

int main() {
    CircularTarget tgt;

    Eigen::VectorXd x_abs(13);
    x_abs.setZero();
    x_abs(0) = 0.0;
    x_abs(1) = 0.0;
    x_abs(2) = 5.0;
    x_abs(6) = 1.0; // upright quaternion

    const double t0 = 0.0;
    Eigen::VectorXd x_rel(13);
    x_rel.head(3)      = x_abs.head(3)      - tgt.pos(t0);
    x_rel.segment(3,3) = x_abs.segment(3,3) - tgt.vel(t0);
    x_rel.segment(6,7) = x_abs.segment(6,7);

    QuadrotorMPC::Config cfg;
    cfg.ocp_type = "stateswitch";
    cfg.n_shift = 7;

    QuadrotorMPC mpc(cfg);
    auto result = mpc.solve(x_rel, tgt.accel(t0));

    if (result.success == false) {
        std::cerr << "[test_stateswitch_ocp] solve failed\n";
        return 1;
    }

    if (result.state_trajectory.empty()) {
        std::cerr << "[test_stateswitch_ocp] empty trajectory\n";
        return 1;
    }

    const auto& xN = result.state_trajectory.back();
    double pos_err = xN.head(3).norm();

    std::cout << "[test_stateswitch_ocp] pos_err=" << pos_err << "\n";
    std::cout << "[test_stateswitch_ocp] X_dim=" << xN.size()
              << " U_dim=" << (result.control_trajectory.empty() ? 0 : result.control_trajectory.front().size())
              << " K_count=" << result.feedback_gains.size() << "\n";

    for (size_t i = 0; i < result.feedback_gains.size(); ++i) {
        const auto& K = result.feedback_gains[i];
        if ((K.rows() == 5) && (K.cols() == 14)) {
            continue;
        }
        std::cerr << "[test_stateswitch_ocp] K shape mismatch at i=" << i
                  << " got " << K.rows() << "x" << K.cols() << "\n";
        return 1;
    }

    if (pos_err > 0.1) {
        std::cerr << "[test_stateswitch_ocp] terminal pos error too large: " << pos_err << "\n";
        return 1;
    }

    return 0;
}
