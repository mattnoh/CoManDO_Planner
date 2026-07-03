/// @file test_stc_landing.cpp
/// @brief Standalone (no ROS master) receding-horizon smoke test for the
///        "stc_landing" OCP: descriptor sanity, predictor plumbing, and 5
///        warm-started solves through QuadrotorMPC against a moving circular
///        target. Solves 2+ exercise the IDX_DT / IDX_CTCS_Y warm-start rebase
///        inside StcLandingOCP::create().

#include "planner_core/quadrotor_mpc.hpp"
#include "planner_core/ocp_registry.hpp"
#include "ocp/ocp_stc_landing.hpp"

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

// Synthetic circular target (matches the benchmark's default "rotating" case).
struct CircularTarget {
    Eigen::Vector3d center{0.0, 0.0, 0.5};
    double R = 2.0;
    double omega = 0.4;

    Eigen::Vector3d pos(double t) const {
        const double ph = omega * t;
        return center + Eigen::Vector3d(R * std::cos(ph), R * std::sin(ph), 0.0);
    }
    Eigen::Vector3d vel(double t) const {
        const double ph = omega * t;
        return {-R * omega * std::sin(ph), R * omega * std::cos(ph), 0.0};
    }
    Eigen::Vector3d accel(double t) const {
        const double ph = omega * t;
        return {-R * omega * omega * std::cos(ph), -R * omega * omega * std::sin(ph), 0.0};
    }

    TargetSnapshot snapshot(double t) const {
        TargetSnapshot s;
        s.valid = true;
        s.position = pos(t);
        s.velocity = vel(t);
        s.acceleration = accel(t);
        return s;
    }
};

}  // namespace

int main() {
    using StcLandingOCP::HORIZON;
    using StcLandingOCP::NEX;
    using StcLandingOCP::IDX_DT;
    using StcLandingOCP::IDX_CTCS_Y;
    using StcLandingOCP::CTCS_Y_SCALE;

    try {
        // ── 1. Descriptor / registry sanity ─────────────────────────────────
        const OCPDescriptor& desc = OCPRegistry::getDescriptor("stc_landing");
        require(desc.name == "stc_landing", "descriptor name mismatch");
        require(desc.variable_dt, "stc_landing must be variable-dt");
        require(desc.state_dim == 13, "physical state_dim must be 13");
        require(desc.control_dim == 4, "physical control_dim must be 4");
        require(desc.warm_start == OCPDescriptor::WarmStart::Feedback,
                "stc_landing must use feedback warm start");
        require(desc.dt > 0.0, "descriptor dt must be positive");
        require(static_cast<bool>(desc.sanitize_warm_control),
                "sanitize_warm_control hook must be set");
        require(OCPRegistry::getDescriptor("rh_stc").name == "stc_landing",
                "rh_stc alias must resolve to stc_landing");

        CircularTarget tgt;
        PlannerConfig planner_cfg;

        // prepare_extra round-trip: predictor must carry the snapshot p/v/a.
        {
            const TargetSnapshot snap = tgt.snapshot(1.5);
            auto extra_any = desc.prepare_extra(planner_cfg, 1.5, snap);
            auto extra = std::any_cast<StcLandingOCP::StcLandingExtra>(extra_any);
            require(static_cast<bool>(extra.predictor), "extra must carry predictor");
            require((extra.predictor->predictPos(0.0) - snap.position).norm() < 1e-12,
                    "predictor position must come from the snapshot");
            require((extra.predictor->predictVel(0.0) - snap.velocity).norm() < 1e-12,
                    "predictor velocity must come from the snapshot");
            require((extra.predictor->predictAccel(2.0) - snap.acceleration).norm() < 1e-12,
                    "const-accel predictor must hold snapshot acceleration");
        }

        // ── 2. Receding-horizon smoke through QuadrotorMPC ──────────────────
        QuadrotorMPC::Config mpc_cfg;
        mpc_cfg.ocp_type = "stc_landing";
        mpc_cfg.n_shift = NEX;
        QuadrotorMPC mpc(mpc_cfg);

        // Benchmark default initial condition (relative frame).
        Eigen::VectorXd x_rel0(13);
        x_rel0 << 2.4, 2.0, 2.0, 0.0, -2.0, 0.0,
                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        double t_abs = 0.0;
        Eigen::VectorXd xq(13);  // absolute world state
        xq.head(3)      = tgt.pos(0.0) + x_rel0.head(3);
        xq.segment(3,3) = tgt.vel(0.0) + x_rel0.segment(3,3);
        xq.segment(6,7) = x_rel0.segment(6,7);

        const double initial_rel_dist = x_rel0.head(3).norm();
        double final_rel_dist = initial_rel_dist;
        const int n_solves = 5;

        for (int s = 0; s < n_solves; ++s) {
            const TargetSnapshot snap = tgt.snapshot(t_abs);
            const Eigen::VectorXd x_rel = desc.transform_state(xq, snap);
            auto extra = desc.prepare_extra(planner_cfg, t_abs, snap);

            auto res = mpc.solve(x_rel, snap.acceleration, extra, t_abs);
            require(res.success, "RH solve must succeed");
            require(static_cast<int>(res.state_trajectory.size()) == HORIZON + 1,
                    "state trajectory must have HORIZON+1 nodes");
            require(static_cast<int>(res.control_trajectory.size()) == HORIZON,
                    "control trajectory must have HORIZON nodes");
            require(StcLandingOCP::finiteTrajectory(res.state_trajectory,
                                                    res.control_trajectory),
                    "trajectory must be finite");

            const auto& X = res.state_trajectory;
            require(std::abs(X[0](IDX_DT)) < 1e-9, "IDX_DT must restart at 0 each solve");
            require(std::abs(X[0](IDX_CTCS_Y)) < 1e-9, "IDX_CTCS_Y must restart at 0 each solve");
            for (int k = 0; k < HORIZON; ++k) {
                require(X[k+1](IDX_DT) > X[k](IDX_DT) - 1e-12,
                        "IDX_DT must be nondecreasing along the horizon");
            }
            // The benchmark's terminal-EQ solves leave y_N in the 0..~0.1
            // range on intermediate RH solves (0.0 / 0.0998 / 0.0026 on the
            // first three); assert it stays bounded rather than exactly zero.
            const double y_terminal = std::abs(X.back()(IDX_CTCS_Y)) * CTCS_Y_SCALE;
            require(y_terminal < 1.0,
                    "terminal CT-cSTC accumulator must stay bounded");

            std::cout << "[test_stc_landing] solve " << s
                      << ": " << res.solve_time_ms << " ms, iters=" << res.solve_iters
                      << ", T*=" << X.back()(IDX_DT)
                      << ", y_N=" << y_terminal
                      << ", rel_dist=" << X[0].head(3).norm() << "\n";

            // Execute NEX nodes: advance time by the executed prefix duration
            // and restitch the relative plan onto the TRUE target (benchmark
            // "exec true target" mode).
            const double exec_dt = X[NEX](IDX_DT);
            require(exec_dt > 0.0, "executed prefix must advance time");
            t_abs += exec_dt;
            xq.head(3)      = X[NEX].head(3) + tgt.pos(t_abs);
            xq.segment(3,3) = X[NEX].segment(3,3) + tgt.vel(t_abs);
            xq.segment(6,7) = X[NEX].segment(6,7);
            final_rel_dist = X[NEX].head(3).norm();
        }

        require(final_rel_dist < initial_rel_dist,
                "landing must make progress toward the target across solves");
        // Warm-started solves converge on the target within a few horizons
        // (benchmark reaches terminal pos_err ~0.03 m by the second solve).
        require(mpc.getPrevX().back().head(3).norm() < 0.5,
                "final horizon must terminate near the target");

        std::cout << "[test_stc_landing] PASS: " << n_solves
                  << " warm-started RH solves, rel_dist "
                  << initial_rel_dist << " -> " << final_rel_dist
                  << ", t_abs=" << t_abs << " s\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[test_stc_landing] FAIL: " << e.what() << "\n";
        return 1;
    }
}
