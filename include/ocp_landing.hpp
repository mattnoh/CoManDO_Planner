/// @file ocp_landing.hpp
/// @brief OCP formulation for landing problem (for online replanning).
///
/// Ported from standalone quad_cf.cpp — all feasibility fixes applied:
///   1. DeltaUStageCost    — S penalty on moment slew rate (replaces GenericStageCost)
///   2. MaxMomentConstraint — SOC constraint on [Mx,My,Mz] from allocation matrix
///   3. Q_DIAG / S_DIAG    — synced to verified standalone values
///
/// FMIN = 0.200 N, FMAX = 0.400 N — tightened band; limits |Δvz| ≤ 0.24 m/s/step
/// so Mellinger can track the planned descent profile.

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
//
// FMIN/FMAX tightened to enforce Mellinger jerk limit:
//   net a_z range = (FMIN - mg)/m .. (FMAX - mg)/m
//                = (0.200 - 0.265)/0.027 .. (0.400 - 0.265)/0.027
//                = -2.41 .. +5.00 m/s²
//   |Δvz|_max = 2.41 * DT = 0.24 m/s/step  — Mellinger-trackable
// With FMIN=0.08 the solver could plan Δvz = 6.5 m/s² → 0.65 m/s/step,
// which Mellinger cannot track and which drives the oscillating-solve failure.
//
const double FMIN       = 0.200;   // N — tightened lower bound (~75 % hover)
const double FMAX       = 0.400;   // N — symmetric upper bound around hover (0.265 N)
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

// ── Solver parameters ─────────────────────────────────────────────────────────
const double SOLVER_REG1_MIN  = 1e-6;
const double SOLVER_REG2_MIN  = 1e-2;
const double SOLVER_MU_MUL    = 0.1;
const double SOLVER_RHO       = 1.0;
const double SOLVER_RHO_MUL   = 10.0;
const double SOLVER_TOLERANCE = 5.0;
const int    SOLVER_MAX_ITER  = 300;
const double SOLVER_RHOT      = 1.0;

// ── Q: running state cost (13×13 diagonal) ────────────────────────────────────
static const Eigen::VectorXd Q_DIAG = (Eigen::VectorXd(13) <<
    2.0, 2.0, 2.0,        // position
    1.0, 1.0, 2.0,        // velocity — vz reduced from 10→2 (S_fz now handles smoothing)
    0.1,                  // qw
    0.1, 0.1, 0.1,        // qx, qy, qz
    0.1, 0.1, 0.1).finished();  // angular rate — reduced: Mellinger owns ω bandwidth

// ── R: running control cost (4×4 diagonal) ────────────────────────────────────
static const Eigen::VectorXd R_DIAG = (Eigen::VectorXd(4) <<
    1e-3,
    1e-4, 1e-4, 1e-4).finished();

// ── S: delta-u (slew-rate) penalty (4×4 diagonal) ─────────────────────────────
// NOTE: In the original create() (13-dim state) this is anchored to u_ref and
// is structurally equivalent to a heavier R — it does NOT couple consecutive
// controls. Use the augmented create_aug() below which reads u_prev from
// state slots 13–16 for genuine consecutive-step coupling.
//
// Jerk limit (Korean report §3, differential flatness):
//   |Δfz| ≤ m*(g)*ω_xy_max*dt = 0.027*9.81*5.0*0.1 ≈ 0.132 N/step
//   S_fz sized so slew cost dominates vz state cost at the limit:
//   S_fz*(0.132)² ≥ Q_vz*(0.1)²  →  S_fz ≥ 1.15 → use 2.0
static const Eigen::VectorXd S_DIAG = (Eigen::VectorXd(4) <<
    2.0,                          // dfz — enforces jerk limit (effective in aug OCP)
    1e-1, 1e-1, 1e-1).finished(); // dMx, dMy, dMz

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
// AugStageCost — stage cost operating on 17-dim augmented state
//
//   J_k = (x[0:13] - x_ref)^T Q (x[0:13] - x_ref)
//       + (u - u_ref)^T R (u - u_ref)
//       + (u - x[13:16])^T S (u - x[13:16])   ← genuine u[k]-u[k-1] slew
//
// S now reads u_prev from x_aug[13:16] rather than a fixed anchor, so the
// DDP backward pass couples consecutive controls through the Riccati recursion.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class AugStageCost : public StageCostBase<Scalar> {
    Eigen::VectorXd x_ref_, u_ref_, Q_diag_, R_diag_, S_diag_;
public:
    AugStageCost(const Eigen::VectorXd& x_ref,
                 const Eigen::VectorXd& u_ref,
                 const Eigen::VectorXd& Q_diag,
                 const Eigen::VectorXd& R_diag,
                 const Eigen::VectorXd& S_diag)
        : x_ref_(x_ref), u_ref_(u_ref),
          Q_diag_(Q_diag), R_diag_(R_diag), S_diag_(S_diag) {}

    Scalar q(const Vector<Scalar>& x_aug, const Vector<Scalar>& u) const override {
        Vector<Scalar> ex = x_aug.head(13) - x_ref_;
        Vector<Scalar> eu = u - u_ref_;
        Vector<Scalar> du = u - x_aug.tail(4);   // u[k] - u[k-1]
        return ex.dot(Q_diag_.asDiagonal() * ex)
             + eu.dot(R_diag_.asDiagonal() * eu)
             + du.dot(S_diag_.asDiagonal() * du);
    }

    Vector<Scalar> qx(const Vector<Scalar>& x_aug,
                      const Vector<Scalar>& u) const override {
        Vector<Scalar> g(17);
        g.head(13) = 2.0 * (Q_diag_.asDiagonal() * (x_aug.head(13) - x_ref_));
        g.tail(4)  = -2.0 * (S_diag_.asDiagonal() * (u - x_aug.tail(4)));
        return g;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x_aug,
                      const Vector<Scalar>& u) const override {
        return 2.0 * (R_diag_.asDiagonal() * (u - u_ref_))
             + 2.0 * (S_diag_.asDiagonal() * (u - x_aug.tail(4)));
    }

    Matrix<Scalar> qxx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(17, 17);
        H.block(0,  0,  13, 13) = (2.0 * Q_diag_).asDiagonal();
        H.block(13, 13, 4,  4)  = (2.0 * S_diag_).asDiagonal();
        return H;
    }

    Matrix<Scalar> quu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return (2.0 * (R_diag_ + S_diag_)).asDiagonal();
    }

    // qxu: dim_x × dim_u = 17×4
    // Only nonzero: ∂²/∂(u_prev)∂u of S‖u-u_prev‖² = -2S at rows 13:16
    Matrix<Scalar> qxu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(17, 4);
        H.block(13, 0, 4, 4) = (-2.0 * S_diag_).asDiagonal();
        return H;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// AugTerminalCost — terminal cost for 17-dim state (u_prev slots zero weight)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class AugTerminalCost : public TerminalCostBase<Scalar> {
    Eigen::VectorXd x_ref_, P_diag_;
public:
    AugTerminalCost(const Eigen::VectorXd& x_ref, const Eigen::VectorXd& P_diag)
        : x_ref_(x_ref), P_diag_(P_diag) {}

    Scalar p(const Vector<Scalar>& x_aug) const override {
        Vector<Scalar> e = x_aug.head(13) - x_ref_;
        return e.dot(P_diag_.asDiagonal() * e);
    }

    Vector<Scalar> px(const Vector<Scalar>& x_aug) const override {
        Vector<Scalar> g(17);
        g.head(13) = 2.0 * (P_diag_.asDiagonal() * (x_aug.head(13) - x_ref_));
        g.tail(4).setZero();
        return g;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>&) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(17, 17);
        H.block(0, 0, 13, 13) = (2.0 * P_diag_).asDiagonal();
        return H;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// AugConstraintWrapper — pads any 13-dim constraint's cx to 17 columns.
//
// All physical constraints (thrust, moment, glideslope, tilt) only use
// x[0:12] and u. This wrapper appends zero columns for u_prev (indices 13-16).
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class AugConstraintWrapper : public StageConstraintBase<Scalar> {
    std::shared_ptr<StageConstraintBase<Scalar>> inner_;
public:
    // Pass constraint_type and dim_c explicitly to avoid accessing protected members.
    // These match the inner constraint's values — copy them from the concrete instance
    // before wrapping, e.g.:
    //   auto inner = std::make_shared<GlideslopeConstraint<double>>();
    //   auto wrapped = std::make_shared<AugConstraintWrapper<double>>(
    //       inner, ConstraintType::SOC, 3);
    AugConstraintWrapper(std::shared_ptr<StageConstraintBase<Scalar>> inner,
                         ConstraintType ctype, int dim)
        : inner_(inner)
    {
        this->constraint_type = ctype;
        this->dim_c           = dim;
    }

    Vector<Scalar> c(const Vector<Scalar>& x_aug,
                     const Vector<Scalar>& u) const override {
        return inner_->c(x_aug.head(13), u);
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x_aug,
                      const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(this->dim_c, 17);
        J.block(0, 0, this->dim_c, 13) = inner_->cx(x_aug.head(13), u);
        return J;
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x_aug,
                      const Vector<Scalar>& u) const override {
        return inner_->cu(x_aug.head(13), u);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// make_x_aug — build a 17-dim augmented initial state
//
//   x_aug = [ x_physical(13), u_prev(4) ]
//
//   On cold start:  u_prev = make_u_ref()  (hover)
//   On warm re-solve: u_prev = prev_U_[0] (last executed control)
// ─────────────────────────────────────────────────────────────────────────────
static Eigen::VectorXd make_x_aug(const Eigen::VectorXd& x13,
                                   const Eigen::VectorXd& u_prev)
{
    Eigen::VectorXd x_aug(17);
    x_aug.head(13) = x13;
    x_aug.tail(4)  = u_prev;
    return x_aug;
}

// ─────────────────────────────────────────────────────────────────────────────
// create_aug — factory for augmented (17-dim state) OCP
//
// Use this instead of create() for Mellinger-tracked RH operation.
// The S slew penalty now genuinely couples u[k] to u[k-1] within the horizon.
//
// @param current_state_aug  17-dim state built with make_x_aug()
// @param prev_U             previous horizon controls for warm-start seeding
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create_aug(
    const Eigen::VectorXd& current_state_aug,
    const Eigen::VectorXd& /* terminal_state */,
    const std::vector<Eigen::VectorXd>& prev_U = {})
{
    auto prob = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    // Augmented dynamics (17-dim)
    auto dyn = std::make_shared<Quad6DOFAug<double>>();
    dyn->setMass(MASS);
    dyn->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
    dyn->setJb(INERTIA);
    dyn->setDt(DT);
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageDynamics(i, dyn);

    const Eigen::VectorXd x_ref = make_x_ref();
    const Eigen::VectorXd u_ref = make_u_ref();

    // AugStageCost: reads u_prev from x_aug[13:16] — no fixed anchor needed
    auto stage_cost = std::make_shared<AugStageCost<double>>(
        x_ref, u_ref, Q_DIAG, R_DIAG, S_DIAG);
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageCost(i, stage_cost);

    prob->setTerminalCost(std::make_shared<AugTerminalCost<double>>(x_ref, P_DIAG));

    // Wrap all physical constraints for 17-dim state.
    // Pass ConstraintType and dim_c explicitly (protected in base class).
    auto gs_inner   = std::make_shared<GlideslopeConstraint<double>>(GLIDESLOPE);
    auto tc_inner   = std::make_shared<TiltConeConstraint<double>>(TILT_CONE);
    auto mt_inner   = std::make_shared<MaxThrustConstraint<double>>(FMAX);
    auto fmin_inner = std::make_shared<MinThrustConstraint<double>>();
    auto mm_inner   = std::make_shared<MaxMomentConstraint<double>>();

    auto gs   = std::make_shared<AugConstraintWrapper<double>>(gs_inner,   ConstraintType::SOC, 3);
    auto tc   = std::make_shared<AugConstraintWrapper<double>>(tc_inner,   ConstraintType::SOC, 3);
    auto mt   = std::make_shared<AugConstraintWrapper<double>>(mt_inner,   ConstraintType::NO,  1);
    auto fmin = std::make_shared<AugConstraintWrapper<double>>(fmin_inner, ConstraintType::NO,  1);
    auto mm   = std::make_shared<AugConstraintWrapper<double>>(mm_inner,   ConstraintType::SOC, 4);

    for (int i = 0; i < HORIZON; ++i) {
        prob->addStageConstraint(i, gs);
        prob->addStageConstraint(i, tc);
        prob->addStageConstraint(i, mt);
        prob->addStageConstraint(i, fmin);
        prob->addStageConstraint(i, mm);
    }

    prob->setInitialState(0, current_state_aug);

    // Warm-start controls
    Eigen::VectorXd u0(4);
    u0 << MASS * 9.81, 0.0, 0.0, 0.0;
    for (int i = 0; i < HORIZON; ++i) {
        const Eigen::VectorXd& ui = (prev_U.size() > (size_t)i) ? prev_U[i] : u0;
        prob->setInitialControl(i, ui);
    }

    return prob;
}

// ─────────────────────────────────────────────────────────────────────────────
// create — delegates to create_aug (augmented 17-dim state).
//
// ocp_registry.hpp calls: LandingOCP::create(current_state, terminal_state, prev_U)
// This function accepts that signature and builds the augmented initial state
// automatically, so no changes to ocp_registry or quadrotor_mpc are required.
//
// u_prev for the augmented state: prev_U[0] if available, hover otherwise.
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& terminal_state,
    const std::vector<Eigen::VectorXd>& prev_U = {})
{
    const Eigen::VectorXd u_prev = prev_U.empty() ? make_u_ref() : prev_U[0];
    const Eigen::VectorXd x_aug  = make_x_aug(current_state, u_prev);
    return create_aug(x_aug, terminal_state, prev_U);
}

} // namespace LandingOCP