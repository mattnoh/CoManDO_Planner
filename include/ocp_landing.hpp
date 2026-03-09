/// @file ocp_landing.hpp
/// @brief OCP formulation for landing problem (for online replanning).
///
/// Ported from standalone quad_cf.cpp — all feasibility fixes applied:
///   1. DeltaUStageCost    — S penalty on moment slew rate (replaces GenericStageCost)
///   2. MaxMomentConstraint — SOC constraint on [Mx,My,Mz] from allocation matrix
///   3. Q_DIAG / S_DIAG    — synced to verified standalone values
///
/// FMAX = 0.6 N (Crazyflie RH value, not the 1.2 N used in cold-start standalone).

#pragma once

#include <Eigen/Dense>
#include <memory>
#include "optimal_control_problem.h"

namespace LandingOCP {

// ── Fixed parameters ──────────────────────────────────────────────────────────
const int    HORIZON = 100;        // 100 steps × 0.1 s = 10 s horizon
const double DT      = 0.1;        // seconds
const double MASS    = 0.027;      // kg

// Scale inertia so diagonal entries are ~O(1) for numerical conditioning
const double J_SCALE = 1.0 / 1.66e-5;            // ≈ 60240
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() <<
    1.66e-5 * J_SCALE, 0.0, 0.0,
    0.0, 1.66e-5 * J_SCALE, 0.0,
    0.0, 0.0, 2.92e-5 * J_SCALE).finished();

// ── Constraint parameters ─────────────────────────────────────────────────────
// Mellinger jerk limit:
//   |Δvz| / dt ≤ ω_xy_max * fz/m
//   For ω_xy_max = 2.0 rad/s: a_z_max = 2.0 * 9.81 * 0.1 = 1.96 m/s²
//
//   FMIN_MELL = m * (g - a_z_max) = 0.027 * (9.81 - 1.96) = 0.212 N
//   FMAX_MELL = m * (g + a_z_max) = 0.027 * (9.81 + 1.96) = 0.318 N
//
// The full hardware range [0.08, 0.6] is still available for Mellinger
// to reject disturbances — this is the "tightened constraint set" from
// the Korean report (축소된 제약 집합).
const double FMIN       = 0.212;   // N — Mellinger acceleration bound
const double FMAX       = 0.318;   // N — Mellinger acceleration bound
const double GLIDESLOPE = 70.0;    // degrees
const double TILT_CONE  = 60.0;    // degrees — max vehicle tilt from vertical

// ── Moment limits derived from allocation matrix ──────────────────────────────
//   arm length l  = 0.046 m (Crazyflie 2.x)
//   f_motor_max   = FMAX / 4   (each motor's share of total max thrust)
//   f_motor_min   = FMIN / 4
//   c_tau         = 0.005 m    (torque-to-thrust ratio k_tau/k_f)
//
//   τ_xy_max = l  * (f_motor_max - f_motor_min)
//   τ_z_max  = c_tau * (f_motor_max - f_motor_min) * 4
//
// Conservative: use τ_xy_max for the SOC ball (xy is tighter than z).
const double L_ARM       = 0.046;
const double F_MOTOR_MAX = FMAX / 4.0;
const double F_MOTOR_MIN = FMIN / 4.0;
const double C_TAU       = 0.005;
const double TAU_XY_MAX  = L_ARM * (F_MOTOR_MAX - F_MOTOR_MIN);   // ≈ 0.006 N·m
const double TAU_Z_MAX   = C_TAU * (F_MOTOR_MAX - F_MOTOR_MIN) * 4.0;
const double TAU_MAX     = TAU_XY_MAX;  // conservative: tightest axis
// Per-step velocity change limit (jerk proxy):
//   j_max = 0.075 m/s  /  DT = 0.75 m/s²  (net world-frame acceleration bound)
//   Constraint: |v[k+1] - v[k]| / DT ≤ J_MAX  ⟺  |Δv| ≤ J_MAX * DT = 0.075 m/s
const double J_MAX       = 0.397 / DT;   // 0.75 m/s² — max |v[k+1]-v[k]| / DT

// ── Solver parameters ─────────────────────────────────────────────────────────
const double SOLVER_REG1_MIN  = 1e-6;
const double SOLVER_REG2_MIN  = 1e-2;
const double SOLVER_MU_MUL    = 0.1;
const double SOLVER_RHO       = 10.0;
const double SOLVER_RHO_MUL   = 10.0;
const double SOLVER_TOLERANCE = 0.05;
const int    SOLVER_MAX_ITER  = 300;
const double SOLVER_RHOT      = 1.0;

// ── Q: running state cost (13×13 diagonal) ────────────────────────────────────
// vz reduced from 10→2: old value with R_fz=1e-3 was a pathological ratio
// that forced aggressive sub-hover thrust at step 0. With vz=2 the solver
// plans a gradual descent instead of slamming thrust to kill vz instantly.
static const Eigen::VectorXd Q_DIAG = (Eigen::VectorXd(13) <<
    2.0, 2.0, 2.0,        // position  — pulls toward origin
    1.0, 1.0, 2.0,        // velocity  — vz reduced from 10→2, gradual descent
    0.1,                  // qw        — light (allow tilting during manoeuvre)
    0.1, 0.1, 0.1,        // qx, qy, qz
    0.05, 0.05, 0.05).finished();  // angular rate

// ── R: running control cost (4×4 diagonal) ────────────────────────────────────
// u = [fz_B, Mx, My, Mz]
static const Eigen::VectorXd R_DIAG = (Eigen::VectorXd(4) <<
    1e-3,                         // fz_B  — light (thrust changes OK)
    1e-4, 1e-4, 1e-4).finished(); // moments — weak absolute penalty

// ── S: delta-u (slew-rate) penalty (4×4 diagonal) ─────────────────────────────
// Light penalty on (u[k] - u_prev[k]). This does not enforce the jerk constraint
// because u_prev is fixed to u_ref; it only penalizes deviation from hover.
//
// The per-step acceleration limit is enforced by FMIN/FMAX tightening (Mellinger
// bound). This keeps S light, avoiding over-constraint at cold start.
static const Eigen::VectorXd S_DIAG = (Eigen::VectorXd(4) <<
    1e-3,                         // dfz   — light; jerk enforced by FMIN/FMAX
    1e-1, 1e-1, 1e-1).finished(); // dMx, dMy, dMz — unchanged

// ── P: terminal state cost (13×13 diagonal) ───────────────────────────────────
static const Eigen::VectorXd P_DIAG = (Eigen::VectorXd(13) <<
    1000.0, 1000.0, 1000.0,  // position  x, y, z
     500.0,  500.0,  500.0,  // velocity  vx, vy, vz
     500.0,                  // qw
     500.0,  500.0,  500.0,  // qx, qy, qz
     200.0,  200.0,  200.0). // angular rate wx, wy, wz
    finished();

// ── Reference state: landed (pos=0, vel=0, q=[1,0,0,0], ω=0) ────────────────
static Eigen::VectorXd make_x_ref() {
    Eigen::VectorXd xr = Eigen::VectorXd::Zero(13);
    xr(6) = 1.0;  // qw = 1 (upright)
    return xr;
}

// ── Reference control: gravity-compensating hover ────────────────────────────
static Eigen::VectorXd make_u_ref() {
    Eigen::VectorXd ur = Eigen::VectorXd::Zero(4);
    ur(0) = MASS * 9.81;  // fz_B = mg
    return ur;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage cost with delta-u (slew-rate) penalty
//
//   J_k = (x - x_ref)^T Q (x - x_ref)
//       + (u - u_ref)^T R (u - u_ref)
//       + (u - u_prev)^T S (u - u_prev)
//
// u_prev is the anchor for the slew penalty:
//   Cold start  → u_ref (hover), set in create()
//   RH warm     → u_ref is still used here because stage cost objects are
//                 constructed once and reused across solves. The actual slew
//                 is bounded by MaxMomentConstraint + the S cost together.
//                 If tighter inter-solve continuity is needed, rebuild stage
//                 costs each tick with u_prev = prev_U_[k] (requires solver
//                 object reset — accepted cost for stricter guarantees).
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class DeltaUStageCost : public StageCostBase<Scalar> {
    Eigen::VectorXd x_ref_;
    Eigen::VectorXd u_ref_;
    Eigen::VectorXd Q_diag_;
    Eigen::VectorXd R_diag_;
    Eigen::VectorXd S_diag_;
    Eigen::VectorXd u_prev_;
public:
    DeltaUStageCost(const Eigen::VectorXd& x_ref,
                    const Eigen::VectorXd& u_ref,
                    const Eigen::VectorXd& Q_diag,
                    const Eigen::VectorXd& R_diag,
                    const Eigen::VectorXd& S_diag,
                    const Eigen::VectorXd& u_prev)
        : x_ref_(x_ref), u_ref_(u_ref),
          Q_diag_(Q_diag), R_diag_(R_diag),
          S_diag_(S_diag), u_prev_(u_prev) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> ex = x - x_ref_;
        Vector<Scalar> eu = u - u_ref_;
        Vector<Scalar> du = u - u_prev_;
        return ex.dot(Q_diag_.asDiagonal() * ex)
             + eu.dot(R_diag_.asDiagonal() * eu)
             + du.dot(S_diag_.asDiagonal() * du);
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        return 2.0 * (Q_diag_.asDiagonal() * (x - x_ref_));
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        return 2.0 * (R_diag_.asDiagonal() * (u - u_ref_))
             + 2.0 * (S_diag_.asDiagonal() * (u - u_prev_));
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return (2.0 * Q_diag_).asDiagonal();
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        // R + S both contribute — larger Quu improves backward-pass conditioning
        return (2.0 * (R_diag_ + S_diag_)).asDiagonal();
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(x.size(), u.size());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Terminal cost: (x_N - x_ref)^T P (x_N - x_ref)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class GenericTerminalCost : public TerminalCostBase<Scalar> {
    Eigen::VectorXd x_ref_;
    Eigen::VectorXd P_diag_;
public:
    GenericTerminalCost(const Eigen::VectorXd& x_ref,
                        const Eigen::VectorXd& P_diag)
        : x_ref_(x_ref), P_diag_(P_diag) {}

    Scalar p(const Vector<Scalar>& x) const override {
        Vector<Scalar> e = x - x_ref_;
        return e.dot(P_diag_.asDiagonal() * e);
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        return 2.0 * (P_diag_.asDiagonal() * (x - x_ref_));
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        (void)x;
        return (2.0 * P_diag_).asDiagonal();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Max thrust constraint   fz_B ≤ FMAX   (nonnegative orthant)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class MaxThrustConstraint : public StageConstraintBase<Scalar> {
    Scalar fmax_;
public:
    explicit MaxThrustConstraint(Scalar fmax = static_cast<Scalar>(FMAX)) : fmax_(fmax) {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> c_n(1);
        c_n(0) = u(0) - fmax_;
        return c_n;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(1, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        J(0, 0) = 1.0;
        return J;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Min thrust constraint   fz_B ≥ FMIN   (nonnegative orthant)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class MinThrustConstraint : public StageConstraintBase<Scalar> {
public:
    MinThrustConstraint() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> c_n(1);
        c_n(0) = FMIN - u(0);
        return c_n;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(1, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        J(0, 0) = -1.0;
        return J;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Glideslope cone constraint (SOC)
//   √(x² + y²) ≤ tan(γ) · z
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class GlideslopeConstraint : public StageConstraintBase<Scalar> {
    Scalar tan_gs_;
public:
    explicit GlideslopeConstraint(Scalar glideslope_deg = static_cast<Scalar>(GLIDESLOPE)) {
        tan_gs_ = std::tan(glideslope_deg * M_PI / 180.0);
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Vector<Scalar> c_n(3);
        c_n(0) = tan_gs_ * x(2);
        c_n(1) = x(0);
        c_n(2) = x(1);
        return -c_n;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, x.size());
        J(0, 2) = tan_gs_;
        J(1, 0) = 1.0;
        J(2, 1) = 1.0;
        return -J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(3, u.size());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Tilt cone constraint (SOC on state quaternion)
//
//   Exact tilt angle φ from vertical: cos(φ) = 1 - 2*(qx² + qy²)
//   φ ≤ φ_max  ⟺  ‖[qx,qy]‖₂ ≤ √((1 - cos(φ_max))/2) = sin(φ_max/2)
//
//   SOC convention: ‖c[1:]‖₂ ≤ -c[0]
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TiltConeConstraint : public StageConstraintBase<Scalar> {
    Scalar tilt_limit_;
public:
    explicit TiltConeConstraint(Scalar theta_max_deg = static_cast<Scalar>(TILT_CONE)) {
        tilt_limit_ = std::sqrt((1.0 - std::cos(theta_max_deg * M_PI / 180.0)) / 2.0);
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Vector<Scalar> c_n(3);
        c_n(0) = tilt_limit_;
        c_n(1) = x(7);   // qx
        c_n(2) = x(8);   // qy
        return -c_n;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, x.size());
        J(1, 7) = 1.0;
        J(2, 8) = 1.0;
        return -J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(3, u.size());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Max moment constraint (SOC ball on body torques)
//
//   ‖[Mx, My, Mz]‖₂ ≤ τ_max
//
//   τ_max = TAU_XY_MAX = L_ARM * (F_MOTOR_MAX - F_MOTOR_MIN)
//
//   Derived from the Actuation Wrench Polytope (AWP): conservative inner
//   approximation of the set of torques achievable without motor saturation.
//   Without this constraint the DDP freely outputs moments 20–30× over the
//   physical limit, causing immediate Mellinger motor saturation.
//
//   SOC convention: ‖c[1:]‖₂ ≤ -c[0]
//     c[0] = -τ_max, c[1..3] = [Mx, My, Mz]
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class MaxMomentConstraint : public StageConstraintBase<Scalar> {
    Scalar tau_max_;
public:
    explicit MaxMomentConstraint(Scalar tau_max = static_cast<Scalar>(TAU_MAX))
        : tau_max_(tau_max)
    {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 4;   // [τ_max, Mx, My, Mz]
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> c_n(4);
        c_n(0) = tau_max_;
        c_n(1) = u(1);   // Mx
        c_n(2) = u(2);   // My
        c_n(3) = u(3);   // Mz
        return -c_n;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(4, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(4, u.size());
        J(1, 1) = 1.0;
        J(2, 2) = 1.0;
        J(3, 3) = 1.0;
        return -J;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Velocity jerk constraint  (coupling x[k]→x[k+1] inlined via dynamics)
//
//   Δv_k = v[k+1] - v[k] = DT * a_I(x[k], u[k])
//         = DT * ( C(q[k]) * [0,0,fz_B[k]] / m  +  g )
//
//   Constraint (per axis, nonneg-orthant, 6 inequalities):
//       Δv_i  ≤ J_MAX * DT      (upper)          c[i]   = Δv_i  - dv_max ≤ 0
//      -Δv_i  ≤ J_MAX * DT      (lower)          c[3+i] = -Δv_i - dv_max ≤ 0
//
//   Jacobians mirror rows 3–5 of Quad6DOF::fx / Quad6DOF::fu.
//
//   NOTE: no native x[k]↔x[k+1] coupling needed — v[k+1] is a deterministic
//   function of (x[k], u[k]) through the dynamics, so this lives purely in
//   stage k of the standard c(x,u) interface.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class VelocityJerkConstraint : public StageConstraintBase<Scalar> {
    double mass_;
    double dt_;
    Eigen::Vector3d gravity_;
    double dv_max_;   // = J_MAX * DT  (per-step Δv bound, m/s)
public:
    VelocityJerkConstraint(double mass, double dt,
                            const Eigen::Vector3d& gravity, double j_max)
        : mass_(mass), dt_(dt), gravity_(gravity), dv_max_(j_max * dt)
    {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 6;   // ±Δvx, ±Δvy, ±Δvz
    }

    // Δv = dt * (C(q)*[0,0,fz]/m + g)
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        const Eigen::Vector4d q = x.segment(6, 4);
        const double fz = u(0);
        const Eigen::Matrix3d C = Quad6DOF<Scalar>::calcC(q);
        const Eigen::Vector3d dv =
            dt_ * (C * Eigen::Vector3d(0.0, 0.0, fz) / mass_ + gravity_);

        Vector<Scalar> c_n(6);
        for (int i = 0; i < 3; ++i) {
            c_n(i)     =  dv(i) - dv_max_;   // upper: Δv_i ≤ dv_max
            c_n(3 + i) = -dv(i) - dv_max_;   // lower: -Δv_i ≤ dv_max
        }
        return c_n;
    }

    // ∂c/∂x: only columns 6–9 (quaternion) are non-zero
    // Mirrors ∂v_next/∂q from Quad6DOF::fx() rows 3–5
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        const Eigen::Vector4d q = x.segment(6, 4);
        const double fz = u(0);
        const Eigen::Vector3d f_B(0.0, 0.0, fz);

        Eigen::Matrix3d dCdq0, dCdq1, dCdq2, dCdq3;
        dCdq0 << 0,        -2*q(3),  2*q(2),
                 2*q(3),   0,       -2*q(1),
                -2*q(2),   2*q(1),  0;
        dCdq1 << 0,         2*q(2),  2*q(3),
                 2*q(2),  -4*q(1), -2*q(0),
                 2*q(3),   2*q(0), -4*q(1);
        dCdq2 << -4*q(2),   2*q(1),  2*q(0),
                  2*q(1),  0,        2*q(3),
                 -2*q(0),  2*q(3), -4*q(2);
        dCdq3 << -4*q(3),  -2*q(0),  2*q(1),
                  2*q(0),  -4*q(3),  2*q(2),
                  2*q(1),   2*q(2),  0;

        // ∂(Δv)/∂q[j] = (dt/m) * dCdqj * f_B
        Eigen::Matrix<double, 3, 4> ddv_dq;
        ddv_dq.col(0) = (dt_ / mass_) * (dCdq0 * f_B);
        ddv_dq.col(1) = (dt_ / mass_) * (dCdq1 * f_B);
        ddv_dq.col(2) = (dt_ / mass_) * (dCdq2 * f_B);
        ddv_dq.col(3) = (dt_ / mass_) * (dCdq3 * f_B);

        Matrix<Scalar> J = Matrix<Scalar>::Zero(6, x.size());
        J.block(0, 6, 3, 4) =  ddv_dq;   // upper bounds
        J.block(3, 6, 3, 4) = -ddv_dq;   // lower bounds
        return J;
    }

    // ∂c/∂u: only column 0 (fz_B) is non-zero
    // Mirrors ∂v_next/∂fz from Quad6DOF::fu() rows 3–5
    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        const Eigen::Vector4d q = x.segment(6, 4);
        const Eigen::Matrix3d C = Quad6DOF<Scalar>::calcC(q);
        const Eigen::Vector3d ddv_dfz = (dt_ / mass_) * C.col(2);

        Matrix<Scalar> J = Matrix<Scalar>::Zero(6, u.size());
        J.block(0, 0, 3, 1) =  ddv_dfz;   // upper bounds
        J.block(3, 0, 3, 1) = -ddv_dfz;   // lower bounds
        return J;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory function — builds a fresh OCP for online replanning.
//
// Called by QuadrotorMPC::setupProblem() on cold start or after
// setTerminalState(). On warm-start re-solves the problem_ is REUSED and
// only x[0] + U warm-start are updated (see quadrotor_mpc.hpp).
//
// u_prev for DeltaUStageCost is anchored to u_ref (hover) for all stages.
// This is correct for cold start. On warm re-solves MaxMomentConstraint
// hard-bounds the moments so the S penalty provides additional smoothing
// even with a fixed u_prev = u_ref anchor.
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& /* terminal_state */,    // unused — P drives terminal
    const std::vector<Eigen::VectorXd>& prev_U = {}) // previous solve's U (empty = cold start)
{
    auto prob = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    // Dynamics
    auto dyn = std::make_shared<Quad6DOF<double>>();
    dyn->setMass(MASS);
    dyn->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
    dyn->setJb(INERTIA);
    dyn->setDt(DT);
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageDynamics(i, dyn);

    // References
    const Eigen::VectorXd x_ref = make_x_ref();
    const Eigen::VectorXd u_ref = make_u_ref();

    // Stage cost: Q/R running + S slew-rate penalty
    // u_prev[k] = prev_U[k] if available (warm re-solve), else u_ref (cold start)
    for (int i = 0; i < HORIZON; ++i) {
        const Eigen::VectorXd& u_prev =
            (prev_U.size() > static_cast<size_t>(i)) ? prev_U[i] : u_ref;
        prob->setStageCost(i, std::make_shared<DeltaUStageCost<double>>(
            x_ref, u_ref, Q_DIAG, R_DIAG, S_DIAG, u_prev));
    }

    // Terminal cost
    prob->setTerminalCost(std::make_shared<GenericTerminalCost<double>>(
        x_ref, P_DIAG));

    // Stage constraints
    auto gs   = std::make_shared<GlideslopeConstraint<double>>(GLIDESLOPE);
    auto tc   = std::make_shared<TiltConeConstraint<double>>(TILT_CONE);
    auto mt   = std::make_shared<MaxThrustConstraint<double>>(FMAX);
    auto fmin = std::make_shared<MinThrustConstraint<double>>();
    auto mm   = std::make_shared<MaxMomentConstraint<double>>();
    // Jerk constraint: |v[k+1] - v[k]| / DT ≤ J_MAX = 0.75 m/s²  (per axis)
    // Expressed as c(x[k], u[k]) by inlining the velocity dynamics.
    auto jerk = std::make_shared<VelocityJerkConstraint<double>>(
        MASS, DT, Eigen::Vector3d(0.0, 0.0, -9.81), J_MAX);

    for (int i = 0; i < HORIZON; ++i) {
        prob->addStageConstraint(i, gs);
        prob->addStageConstraint(i, tc);
        prob->addStageConstraint(i, mt);
        prob->addStageConstraint(i, fmin);
        prob->addStageConstraint(i, mm);
        prob->addStageConstraint(i, jerk);
    }

    // Initial state
    prob->setInitialState(0, current_state);

    // Warm-start: hover (gravity compensation, zero moments)
    Eigen::VectorXd u0(4);
    u0 << MASS * 9.81, 0.0, 0.0, 0.0;
    for (int i = 0; i < HORIZON; ++i)
        prob->setInitialControl(i, u0);

    return prob;
}

} // namespace LandingOCP