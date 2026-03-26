#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace target_models {

struct CircularTarget {
    Eigen::Vector3d center{0.0, 0.0, 1.5};
    double R = 2.0;
    double omega = 0.4;
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

    Eigen::VectorXd state(double t) const {
        Eigen::VectorXd s(6);
        s.head(3) = pos(t);
        s.tail(3) = vel(t);
        return s;
    }
};

inline CircularTarget getDefaultCircularTarget() {
    return CircularTarget{};
}

} // namespace target_models
