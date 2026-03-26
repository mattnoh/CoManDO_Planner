#pragma once

#include "circular_target.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cassert>

namespace target_models {

struct TargetAccelBuffer {
    double t_start = 0.0;
    double dt      = 0.05;
    std::vector<Eigen::Vector3d> accels;

    // Call this before solve(). duration should cover worst case horizon.
    void populateFromModel(const CircularTarget& tgt,
                           double start_time, double duration, double step) {
        t_start = start_time;
        dt      = step;
        accels.clear();
        for (double t = start_time; t <= start_time + duration + step; t += step)
            accels.push_back(tgt.accel(t));
    }

    // Pass uniformly spaced vector of accelerations.
    void populateFromSamples(const std::vector<double>& times,
                             const std::vector<Eigen::Vector3d>& accel_samples) {
        assert(!times.empty());
        t_start = times.front();
        dt      = (times.size() > 1) ? (times[1] - times[0]) : 0.05;
        accels  = accel_samples;
    }

    // Linear interpolation
    Eigen::Vector3d getAccel(double t_abs) const {
        if (accels.empty()) return Eigen::Vector3d::Zero();
        double t_rel = t_abs - t_start;
        if (t_rel <= 0.0)  return accels.front();
        int idx = static_cast<int>(t_rel / dt);
        if (idx >= static_cast<int>(accels.size()) - 1) return accels.back();
        double alpha = (t_rel - idx * dt) / dt;
        return accels[idx] * (1.0 - alpha) + accels[idx + 1] * alpha;
    }
};

} // namespace target_models
