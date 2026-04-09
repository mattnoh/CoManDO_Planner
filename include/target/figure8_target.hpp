#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace target_models {

// Gerono figure-8 in the XY plane:
//   x(t) = cx + Ax cos(w t + phi0)
//   y(t) = cy + Ay sin(w t + phi0) cos(w t + phi0)
//   z(t) = cz
struct Figure8Target {
    Eigen::Vector3d center{0.0, 0.0, 0.2};
    double amp_x = 1.0;
    double amp_y = 1.0;
    double omega = 0.1;
    double phi0 = 0.0;

    Eigen::Vector3d pos(double t) const {
        const double ph = omega * t + phi0;
        const double s = std::sin(ph);
        const double c = std::cos(ph);
        return center + Eigen::Vector3d(amp_x * c, amp_y * s * c, 0.0);
    }

    Eigen::Vector3d vel(double t) const {
        const double ph = omega * t + phi0;
        return Eigen::Vector3d(
            -amp_x * omega * std::sin(ph),
            amp_y * omega * std::cos(2.0 * ph),
            0.0);
    }

    Eigen::Vector3d accel(double t) const {
        const double ph = omega * t + phi0;
        return Eigen::Vector3d(
            -amp_x * omega * omega * std::cos(ph),
            -2.0 * amp_y * omega * omega * std::sin(2.0 * ph),
            0.0);
    }

    Eigen::VectorXd state(double t) const {
        Eigen::VectorXd s(6);
        s.head(3) = pos(t);
        s.tail(3) = vel(t);
        return s;
    }
};

inline Figure8Target getDefaultFigure8Target() {
    return Figure8Target{};
}

} // namespace target_models
