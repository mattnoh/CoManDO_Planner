#pragma once

#include "target/circular_target.hpp"
#include "target/figure8_target.hpp"
#include <functional>
#include <Eigen/Dense>

namespace target_config {

// Type-erased wrapper — same interface regardless of which model is active.
struct SyntheticTarget {
    std::function<Eigen::Vector3d(double)> pos;
    std::function<Eigen::Vector3d(double)> vel;
    std::function<Eigen::Vector3d(double)> accel;

    template <typename Model>
    static SyntheticTarget from(Model m) {
        return {
            [m](double t) { return m.pos(t); },
            [m](double t) { return m.vel(t); },
            [m](double t) { return m.accel(t); },
        };
    }
};

// ── Edit here to switch trajectory ──────────────────────────────────────────
inline SyntheticTarget makeTarget()
{
    target_models::CircularTarget m;
    m.center = {0.0, 0.0, 1.0};
    m.R      = 0.7;
    m.omega  = 0.5;
    m.phi0   = 0.0;
    return SyntheticTarget::from(m);

    // To use figure-8 instead, comment the above and uncomment:
    // target_models::Figure8Target m;
    // m.center = {0.0, 0.0, 1.0};
    // m.amp_x  = 0.7;
    // m.amp_y  = 0.5;
    // m.omega  = 0.5;
    // m.phi0   = 0.0;
    // return SyntheticTarget::from(m);
}

} // namespace target_config
