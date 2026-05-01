#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "planner_core/target_snapshot.hpp"

namespace OCPLoggerDefaults {

inline Eigen::VectorXd makeHoverState13D(const Eigen::VectorXd& current_state) {
    Eigen::VectorXd x_hover = (current_state.size() >= 13) 
        ? current_state 
        : Eigen::VectorXd::Zero(13);
    x_hover.segment(3, 3).setZero();
    x_hover(6) = 1.0;
    x_hover.segment(7, 3).setZero();
    x_hover.segment(10, 3).setZero();
    return x_hover;
}

inline std::vector<std::string> getStateHeaders13D() {
    return {
        "x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz",
        "abs_x","abs_y","abs_z","abs_vx","abs_vy","abs_vz","abs_qw","abs_qx","abs_qy","abs_qz","abs_wx","abs_wy","abs_wz",
        "rel_x","rel_y","rel_z","rel_vx","rel_vy","rel_vz","rel_qw","rel_qx","rel_qy","rel_qz","rel_wx","rel_wy","rel_wz",
        "tgt_x","tgt_y","tgt_z","tgt_vx","tgt_vy","tgt_vz"
    };
}

inline std::vector<double> getActualStateRow13D(const Eigen::VectorXd& state, const TargetSnapshot& tgt, const std::string& coord_mode) {
    if (state.size() < 13) return std::vector<double>(45, 0.0);
    std::vector<double> row;
    row.reserve(45);
    Eigen::VectorXd x_abs = state.head(13);
    Eigen::VectorXd x_rel = state.head(13);
    if (coord_mode == "relative") {
        x_abs.segment(0, 3) += tgt.position;
        x_abs.segment(3, 3) += tgt.velocity;
    } else {
        x_rel.segment(0, 3) -= tgt.position;
        x_rel.segment(3, 3) -= tgt.velocity;
    }
    for (int i=0; i<13; ++i) row.push_back(state(i));
    for (int i=0; i<13; ++i) row.push_back(x_abs(i));
    for (int i=0; i<13; ++i) row.push_back(x_rel(i));
    row.push_back(tgt.position.x()); row.push_back(tgt.position.y()); row.push_back(tgt.position.z());
    row.push_back(tgt.velocity.x()); row.push_back(tgt.velocity.y()); row.push_back(tgt.velocity.z());
    return row;
}

// ── State / control name vectors for all_solves.csv ───────────────────────────

/// Raw state names for 13D quad dynamics: [r_I, v_I, q_BI, w_B]
/// Matches Quad6DOF / Quad6DOFVarTime / Quad6DOFVarTimeRelative
inline std::vector<std::string> getStateNames13D() {
    return {"px","py","pz","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz"};
}

/// Raw control names for standard 4D quad control: [fz_B, Mx, My, Mz]
inline std::vector<std::string> getControlNames4D() {
    return {"fz","mx","my","mz"};
}

/// Raw state names for 22D body/target-frame augmented dynamics.
/// Matches Quad6DOFBodyFrameRelative (body-frame) and Quad6DOFTargetFrame (target-frame).
/// x[0:3]=p, x[3:6]=v, x[6:10]=q_NB, x[10:13]=w_B, x[13:16]=OmN, x[16:19]=aT, x[19:22]=bN
inline std::vector<std::string> getStateNames22D() {
    return {"p0","p1","p2","v0","v1","v2","qw","qx","qy","qz",
            "wx","wy","wz","OmN0","OmN1","OmN2","aT0","aT1","aT2",
            "bN0","bN1","bN2"};
}


/// Log headers for actual_state.csv in body-relative mode.
/// Columns: the 13D physical sensor state [p_T_body, v_rel_body, q_NB, omega_B]
/// plus derived world-frame drone position and target reference columns.
inline std::vector<std::string> getStateHeaders23D_Body() {
    return {
        // Raw 13D from /drone/body_relative_odom
        "p0","p1","p2",         // target position in body frame
        "v0","v1","v2",         // relative velocity (drone-target) in body frame
        "qw","qx","qy","qz",   // body attitude quaternion
        "wx","wy","wz",         // body angular rates [rad/s]
        // Derived world-frame drone position: drone_xyz = tgt_xyz - R_WB * p_T_body
        "drone_x","drone_y","drone_z",
        // Convenience: distance from target
        "p_err",
        // Target reference
        "tgt_x","tgt_y","tgt_z","tgt_vx","tgt_vy","tgt_vz"
    };
}

/// Extract actual-state log row from raw 13D sensor state.
/// Works with any state.size() >= 0 — missing fields are written as 0.
/// Output columns match getStateHeaders23D_Body():
///   [p0..p2, v0..v2, qw..qz, wx..wz, drone_x..z, p_err, tgt_x..z, tgt_vx..z]
inline std::vector<double> getActualStateRow23D_Body(
    const Eigen::VectorXd& state,
    const TargetSnapshot& tgt,
    const std::string& /*coord_mode*/)
{
    constexpr int N_PHYS = 13;    // expected sensor state size
    constexpr int N_OUT  = 19;    // p(3) + v(3) + q(4) + w(3) + drone_w(3) + p_err(1) + tgt(6)
    std::vector<double> row;
    row.reserve(N_OUT);

    // Helper: read state element or 0 if out-of-range
    auto s = [&](int i) -> double {
        return (i < (int)state.size()) ? state(i) : 0.0;
    };

    // p_T_body [0:3]
    row.push_back(s(0)); row.push_back(s(1)); row.push_back(s(2));
    // v_rel_body [3:6]
    row.push_back(s(3)); row.push_back(s(4)); row.push_back(s(5));
    // q_NB [6:10]
    row.push_back(s(6)); row.push_back(s(7)); row.push_back(s(8)); row.push_back(s(9));
    // omega_B [10:13]
    row.push_back(s(10)); row.push_back(s(11)); row.push_back(s(12));

    // Derived world-frame drone position: drone_w = tgt_pos - R_WB * p_T_body
    // R_WB from quaternion [qw,qx,qy,qz] = state[6:10]
    {
        const double qw = s(6), qx = s(7), qy = s(8), qz = s(9);
        Eigen::Matrix3d R_WB;
        R_WB << 1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw,
                2*qx*qy + 2*qz*qw,      1 - 2*qx*qx - 2*qz*qz,  2*qy*qz - 2*qx*qw,
                2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,      1 - 2*qx*qx - 2*qy*qy;
        Eigen::Vector3d p_T_b(s(0), s(1), s(2));
        Eigen::Vector3d drone_w = tgt.position - R_WB * p_T_b;
        row.push_back(drone_w(0)); row.push_back(drone_w(1)); row.push_back(drone_w(2));
        row.push_back(p_T_b.norm());  // p_err
    }

    // Target reference
    row.push_back(tgt.position.x()); row.push_back(tgt.position.y()); row.push_back(tgt.position.z());
    row.push_back(tgt.velocity.x()); row.push_back(tgt.velocity.y()); row.push_back(tgt.velocity.z());

    return row;
}

inline Eigen::VectorXd makeHoverState23D_Body(const Eigen::VectorXd& current_state) {
    Eigen::VectorXd x_hover = (current_state.size() >= 22) 
        ? current_state 
        : Eigen::VectorXd::Zero(22);
    x_hover.segment(3, 3).setZero();
    x_hover(6) = 1.0;
    x_hover.segment(7, 3).setZero();
    x_hover.segment(10, 3).setZero();
    return x_hover;
}

} // namespace
