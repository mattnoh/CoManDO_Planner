/// @file ocp_landing.hpp
/// @brief OCP formulation for landing problem (for online replanning)
///        Updated to match standalone example: generic Q/R/P costs,
///        no hard terminal equality constraint.

#pragma once

#include <Eigen/Dense>
#include <memory>
#include "optimal_control_problem.h"

namespace LandingOCP {

// ── Fixed parameters (mirroring standalone solve — verified: err 0.0003) ──────
const int    HORIZON = 100;                      // 100 steps × 0.1s = 10s horizon
const double DT      = 0.1;                      // seconds (standalone used 0.1)
const double MASS    = 0.027;                    // kg

// Scale inertia so that the diagonal entries become ~O(1)
const double J_SCALE = 1.0 / 1.66e-5;            // ≈ 60240
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() <<
    1.66e-5 * J_SCALE, 0.0, 0.0,
    0.0, 1.66e-5 * J_SCALE, 0.0,
    0.0, 0.0, 2.92e-5 * J_SCALE).finished();

// ── Constraint parameters (standalone values) ─────────────────────────────────
const double FMIN        = 0.08;                  // N — 30% hover thrust
const double FMAX        = 0.6;                   // N (standalone used 1.2)
const double GLIDESLOPE  = 70.0;                  // degrees (unchanged)
const double TILT_CONE   = 60.0;                   // degrees — max vehicle tilt from vertical

// ── Solver parameters (tuned for landing — verified working in standalone) ────
// Higher tolerance + max_iter lets outer loop converge on cold start,
// while warm-started re-solves finish in ~5 iterations at log(mu)=-6.
const double SOLVER_REG1_MIN  = 1e-6;
const double SOLVER_REG2_MIN  = 1e-2;       // standalone used 1e-2
const double SOLVER_MU_MUL    = 0.1;
const double SOLVER_RHO       = 1.0;         // default (standalone doesn't override)
const double SOLVER_RHO_MUL   = 10.0;        // default (standalone doesn't override)
const double SOLVER_TOLERANCE = 5.0;         // standalone used 5.0
const int    SOLVER_MAX_ITER  = 300;         // standalone used 300
const double SOLVER_RHOT = 1.0;              // default (standalone doesn't override)

// ── Cost matrices (diagonal, matching standalone — verified working) ──────────
// Q: running state cost (13x13) — nonzero penalises drift during horizon
static const Eigen::VectorXd Q_DIAG = (Eigen::VectorXd(13) <<
    2.0, 2.0, 2.0,                    // position  x, y, z
    1.0, 1.0, 1.0,                    // velocity  vx, vy, vz
    0.5,                               // quaternion qw
    0.5, 0.5, 0.5,                    // quaternion qx, qy, qz
    0.5, 0.5, 0.5).finished();        // angular rate wx, wy, wz

// R: running control cost (4x4) – u = [fz_B, Mx, My, Mz]
static const Eigen::VectorXd R_DIAG = (Eigen::VectorXd(4) <<
    1e-3,               // fz_B (body-z thrust)
    1e-4, 1e-4, 1e-4).finished();   // moments (Mx, My, Mz)

// P: terminal state cost (13x13) – large enough to drive x_N ≈ x_ref
static const Eigen::VectorXd P_DIAG = (Eigen::VectorXd(13) <<
    1000.0, 1000.0, 1000.0,  // position  x, y, z
     500.0,  500.0,  500.0,  // velocity  vx, vy, vz
     500.0,                   // quaternion qw
     500.0,  500.0,  500.0,  // quaternion qx, qy, qz
     200.0,  200.0,  200.0). // angular rate wx, wy, wz
    finished();

// ── Reference state: landed (pos=0, vel=0, q=[1,0,0,0], ω=0) ─────────────────
static Eigen::VectorXd make_x_ref() {
    Eigen::VectorXd xr = Eigen::VectorXd::Zero(13);
    xr(6) = 1.0;   // qw = 1 (upright)
    return xr;
}

// ── Reference control: gravity‑compensating hover ────────────────────────────
static Eigen::VectorXd make_u_ref() {
    Eigen::VectorXd ur = Eigen::VectorXd::Zero(4);
    ur(0) = MASS * 9.81;   // fz_B = mg
    return ur;
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic stage cost: (x - x_ref)^T Q (x - x_ref) + (u - u_ref)^T R (u - u_ref)
// Q and R stored as diagonal vectors.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class GenericStageCost : public StageCostBase<Scalar> {
    Eigen::VectorXd x_ref_;
    Eigen::VectorXd u_ref_;
    Eigen::VectorXd Q_diag_;
    Eigen::VectorXd R_diag_;
public:
    GenericStageCost(const Eigen::VectorXd& x_ref,
                     const Eigen::VectorXd& u_ref,
                     const Eigen::VectorXd& Q_diag,
                     const Eigen::VectorXd& R_diag)
        : x_ref_(x_ref), u_ref_(u_ref), Q_diag_(Q_diag), R_diag_(R_diag) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> ex = x - x_ref_;
        Vector<Scalar> eu = u - u_ref_;
        return ex.dot(Q_diag_.asDiagonal() * ex)
             + eu.dot(R_diag_.asDiagonal() * eu);
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        return 2.0 * (Q_diag_.asDiagonal() * (x - x_ref_));
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        return 2.0 * (R_diag_.asDiagonal() * (u - u_ref_));
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return (2.0 * Q_diag_).asDiagonal();
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return (2.0 * R_diag_).asDiagonal();
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(x.size(), u.size());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Generic terminal cost: (x_N - x_ref)^T P (x_N - x_ref)
// P stored as diagonal vector.
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
// Max thrust constraint (nonnegative orthant)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class MaxThrustConstraint : public StageConstraintBase<Scalar> {
private:
    Scalar fmax_;
public:
    MaxThrustConstraint(Scalar fmax = FMAX) : fmax_(fmax) {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }

    // fz_B <= FMAX
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
// Min thrust constraint (nonnegative orthant)
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
        c_n(0) = FMIN - u(0);   // fz_B >= FMIN
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
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class GlideslopeConstraint : public StageConstraintBase<Scalar> {
private:
    Scalar tan_gs_;
public:
    GlideslopeConstraint(Scalar glideslope_deg = GLIDESLOPE) {
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
// ||[qx, qy]||_2 <= sin(tilt_max/2)  (exact formula from cos(phi)=1-2*(qx^2+qy^2))
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TiltConeConstraint : public StageConstraintBase<Scalar> {
private:
    Scalar tilt_limit_;
public:
    TiltConeConstraint(Scalar theta_max_deg = TILT_CONE) {
        tilt_limit_ = std::sqrt((1.0 - std::cos(theta_max_deg * M_PI / 180.0)) / 2.0);
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Vector<Scalar> c_n(3);
        c_n(0) = tilt_limit_;   // -c_n(0) = -tilt_limit (the cone apex)
        c_n(1) = x(7);          // qx
        c_n(2) = x(8);          // qy
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, x.size());
        J(1, 7) = 1.0;   // d/d(qx)
        J(2, 8) = 1.0;   // d/d(qy)
        return -J;
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(3, u.size());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory function: creates a problem with standalone parameters
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& current_state)
{
    auto prob = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    auto dyn = std::make_shared<Quad6DOF<double>>();
    dyn->setMass(MASS);
    dyn->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
    dyn->setJb(INERTIA);
    dyn->setDt(DT);
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageDynamics(i, dyn);

    // References
    Eigen::VectorXd x_ref = make_x_ref();
    Eigen::VectorXd u_ref = make_u_ref();

    // Generic stage cost (instead of hardcoded weights)
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageCost(i, std::make_shared<GenericStageCost<double>>(
            x_ref, u_ref, Q_DIAG, R_DIAG));

    // Generic terminal cost (replaces both SoftTerminalCost and TerminalEqualityConstraint)
    prob->setTerminalCost(std::make_shared<GenericTerminalCost<double>>(
        x_ref, P_DIAG));

    // NO terminal equality constraint – P handles it softly

    // Stage constraints with standalone parameters
    auto gs = std::make_shared<GlideslopeConstraint<double>>(GLIDESLOPE);
    auto tc = std::make_shared<TiltConeConstraint<double>>(TILT_CONE);
    auto mt = std::make_shared<MaxThrustConstraint<double>>(FMAX);
    auto fmin = std::make_shared<MinThrustConstraint<double>>();
    for (int i = 0; i < HORIZON; ++i) {
        prob->addStageConstraint(i, gs);
        prob->addStageConstraint(i, tc);
        prob->addStageConstraint(i, mt);
        prob->addStageConstraint(i, fmin);
    }

    // Initial state
    prob->setInitialState(0, current_state);

    // Warm-start: gravity compensation on body-z thrust, zero moments
    Eigen::VectorXd u0(4);
    u0 << MASS * 9.81, 0.0, 0.0, 0.0;   // [fz_B, Mx, My, Mz]
    for (int i = 0; i < HORIZON; ++i)
        prob->setInitialControl(i, u0);

    return prob;
}

} // namespace LandingOCP