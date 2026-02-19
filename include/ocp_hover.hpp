#pragma once

#include <Eigen/Dense>
#include <memory>
#include "optimal_control_problem.h"

namespace HoverOCP {

// ── Fixed parameters ─────────────────────────────────────────────────────────
const int    HORIZON = 100;          // 5 s horizon (100 * 0.05)
const double DT      = 0.05;          // seconds
const double MASS    = 0.027;         // kg

// Scaled inertia for better numerical conditioning (diag ≈ 1, 1, 1.76)
const double J_SCALE = 1.0 / 1.66e-5; // ≈ 60240
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() <<
    1.66e-5 * J_SCALE, 0.0, 0.0,
    0.0, 1.66e-5 * J_SCALE, 0.0,
    0.0, 0.0, 2.92e-5 * J_SCALE).finished();

// ── Hover target (hard‑coded to (0,0,1)) ───────────────────────────────────
const Eigen::Vector3d TARGET_POS(0.0, 0.0, 1.0);
const Eigen::Vector3d TARGET_VEL(0.0, 0.0, 0.0);

// ── Cost weights ────────────────────────────────────────────────────────────
const double W_POS_STAGE  = 10.0;      // stage position error
const double W_VEL_STAGE  = 1.0;       // stage velocity error
const double W_ATT_STAGE  = 2.0;       // attitude deviation (from upright)
const double W_THRUST     = 1e-1;      // control effort (thrust)
const double W_MOMENT     = 1e-1;      // control effort (moment)

const double W_POS_TERM   = 500.0;     // terminal position error
const double W_VEL_TERM   = 50.0;      // terminal velocity error

// ── Constraint parameters (optional, but realistic) ────────────────────────
const double FMAX         = 1.2;        // max thrust (N)

// ─────────────────────────────────────────────────────────────────────────────
// Stage cost – hard‑coded target (0,0,1)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class StageCost : public StageCostBase<Scalar> {
public:
    StageCost() = default;  // ignore any passed target

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        // Position and velocity errors relative to target
        Eigen::Vector3d pos = x.template segment<3>(0);
        Eigen::Vector3d vel = x.template segment<3>(3);
        Scalar pos_err = (pos - TARGET_POS).squaredNorm();
        Scalar vel_err = (vel - TARGET_VEL).squaredNorm();

        // Attitude error: penalize deviation from identity quaternion [1,0,0,0]
        Scalar q0 = x(6);
        Eigen::Vector3d qv = x.template segment<3>(7);
        Scalar att_err = qv.squaredNorm() + (1.0 - q0) * (1.0 - q0);

        // Control effort
        Eigen::Vector3d f = u.template segment<3>(0);
        Eigen::Vector3d m = u.template segment<3>(3);
        Scalar thrust_effort = f.squaredNorm();
        Scalar moment_effort = m.squaredNorm();

        return W_POS_STAGE * pos_err
             + W_VEL_STAGE * vel_err
             + W_ATT_STAGE * att_err
             + W_THRUST    * thrust_effort
             + W_MOMENT    * moment_effort;
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());

        // Position gradient
        g.template segment<3>(0) = 2.0 * W_POS_STAGE * (x.template segment<3>(0).eval() - TARGET_POS);

        // Velocity gradient
        g.template segment<3>(3) = 2.0 * W_VEL_STAGE * (x.template segment<3>(3).eval() - TARGET_VEL);

        // Attitude gradient
        Scalar q0 = x(6);
        Eigen::Vector3d qv = x.template segment<3>(7);
        g(6) = -2.0 * W_ATT_STAGE * (1.0 - q0);
        g.template segment<3>(7) = 2.0 * W_ATT_STAGE * qv;

        return g;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> g = Vector<Scalar>::Zero(u.size());
        g.template segment<3>(0) = 2.0 * W_THRUST * u.template segment<3>(0);
        g.template segment<3>(3) = 2.0 * W_MOMENT * u.template segment<3>(3);
        return g;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H.template block<3,3>(0,0) = 2.0 * W_POS_STAGE * Matrix<Scalar>::Identity(3,3);
        H.template block<3,3>(3,3) = 2.0 * W_VEL_STAGE * Matrix<Scalar>::Identity(3,3);
        H(6,6) = 2.0 * W_ATT_STAGE;
        H.template block<3,3>(7,7) = 2.0 * W_ATT_STAGE * Matrix<Scalar>::Identity(3,3);
        return H;
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(u.size(), u.size());
        H.template block<3,3>(0,0) = 2.0 * W_THRUST * Matrix<Scalar>::Identity(3,3);
        H.template block<3,3>(3,3) = 2.0 * W_MOMENT * Matrix<Scalar>::Identity(3,3);
        return H;
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(x.size(), u.size());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Terminal cost – hard‑coded target (0,0,1)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
public:
    TerminalCost() = default;  // ignore any passed target

    Scalar p(const Vector<Scalar>& x) const override {
        Eigen::Vector3d pos = x.template segment<3>(0);
        Eigen::Vector3d vel = x.template segment<3>(3);
        return W_POS_TERM * (pos - TARGET_POS).squaredNorm()
             + W_VEL_TERM * (vel - TARGET_VEL).squaredNorm();
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.template segment<3>(0) = 2.0 * W_POS_TERM * (x.template segment<3>(0).eval() - TARGET_POS);
        g.template segment<3>(3) = 2.0 * W_VEL_TERM * (x.template segment<3>(3).eval() - TARGET_VEL);
        return g;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        (void)x;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H.template block<3,3>(0,0) = 2.0 * W_POS_TERM * Matrix<Scalar>::Identity(3,3);
        H.template block<3,3>(3,3) = 2.0 * W_VEL_TERM * Matrix<Scalar>::Identity(3,3);
        return H;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Max thrust constraint (optional)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class MaxThrustConstraint : public StageConstraintBase<Scalar> {
public:
    MaxThrustConstraint() {
        this->constraint_type = ConstraintType::NO; // nonnegative orthant form
        this->dim_c = 1;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> c_n(1);
        c_n(0) = FMAX - u.template segment<3>(0).norm();
        return -c_n; // transform to ≤ 0 form
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(1, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        auto f = u.template segment<3>(0).eval();
        Scalar nf = f.norm();
        if (nf > 1e-8)
            J.template block<1,3>(0, 0) = -f.transpose() / nf;
        return -J;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory – two arguments for compatibility, but target_state is ignored
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& /*target_state*/)  // ignored – we hover at (0,0,1)
{
    auto prob = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    // Dynamics (shared across all stages)
    auto dyn = std::make_shared<Quad6DOF<double>>();
    dyn->setMass(MASS);
    dyn->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
    dyn->setJb(INERTIA);
    dyn->setDt(DT);
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageDynamics(i, dyn);

    // Cost functions – use hard‑coded target
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageCost(i, std::make_shared<StageCost<double>>());
    prob->setTerminalCost(std::make_shared<TerminalCost<double>>());

    // Optional constraint: max thrust
    auto mt = std::make_shared<MaxThrustConstraint<double>>();
    for (int i = 0; i < HORIZON; ++i)
        prob->addStageConstraint(i, mt);

    // Initial state
    prob->setInitialState(0, current_state);

    // Warm‑start: gravity‑cancelling thrust in body frame
    Eigen::Quaterniond q(current_state(6), current_state(7),
                         current_state(8), current_state(9));
    q.normalize();
    Eigen::Vector3d f0 = q.inverse() * Eigen::Vector3d(0.0, 0.0, MASS * 9.81);
    Eigen::VectorXd u0(6);
    u0 << f0(0), f0(1), f0(2), 0.0, 0.0, 0.0;
    for (int i = 0; i < HORIZON; ++i)
        prob->setInitialControl(i, u0);

    return prob;
}

} // namespace HoverOCP