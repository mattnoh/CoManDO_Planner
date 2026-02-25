#pragma once

#include <Eigen/Dense>
#include <memory>
#include "optimal_control_problem.h"

namespace HoverOCP {

// ── Fixed parameters ─────────────────────────────────────────────────────────
const int    HORIZON = 100;           // 5 s horizon (100 * 0.05)
const double DT      = 0.05;          // seconds
const double MASS    = 0.027;         // kg

const double J_SCALE = 1.0 / 1.66e-5;
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() <<
    1.66e-5 * J_SCALE, 0.0, 0.0,
    0.0, 1.66e-5 * J_SCALE, 0.0,
    0.0, 0.0, 2.92e-5 * J_SCALE).finished();

// ── Cost weights ────────────────────────────────────────────────────────────
const double W_POS_STAGE  = 10.0;
const double W_VEL_STAGE  = 1.0;
const double W_ATT_STAGE  = 2.0;
const double W_THRUST     = 1e-1;
const double W_MOMENT     = 1e-1;

const double W_POS_TERM   = 500.0;
const double W_VEL_TERM   = 50.0;

const double FMAX         = 1.2;

// ─────────────────────────────────────────────────────────────────────────────
// Stage cost — runtime target
//
// PX4 MPC PATTERN:
//   PX4 stores the reference as a parameter (yref) on the problem object and
//   updates it at runtime without recreating the problem. We replicate this by
//   storing target_pos and target_vel as members of StageCost rather than
//   referencing a compile-time constant. create() receives the target_state and
//   passes it into each StageCost instance.
//
//   This means:
//     - create() is called once (cold start or target change), not every tick
//     - The cost landscape is stable across solves for the same target
//     - Changing the target calls setTerminalState() → need_problem_rebuild_=true
//       → create() is called once with the new target on the next solve
//
// WHY THIS MATTERS:
//   The old code had StageCost() = default with TARGET_POS = (0,0,1) baked
//   into the cost function at compile time. The target_state argument to
//   create() was commented out and ignored entirely. This meant:
//     - hover_target_x/y/z ROS parameters had no effect
//     - setTerminalState() had no effect on where the drone tried to go
//     - The drone would only try to reach (0,0,1) regardless of configuration
//     - Starting the drone at z≈1 would produce x[1]≈x[0] — "drone doesn't move"
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class StageCost : public StageCostBase<Scalar> {
private:
    Eigen::Vector3d target_pos_;
    Eigen::Vector3d target_vel_;
public:
    StageCost(const Eigen::Vector3d& target_pos,
              const Eigen::Vector3d& target_vel = Eigen::Vector3d::Zero())
        : target_pos_(target_pos), target_vel_(target_vel) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Eigen::Vector3d pos = x.template segment<3>(0);
        Eigen::Vector3d vel = x.template segment<3>(3);
        Scalar pos_err = (pos - target_pos_).squaredNorm();
        Scalar vel_err = (vel - target_vel_).squaredNorm();

        Scalar q0 = x(6);
        Eigen::Vector3d qv = x.template segment<3>(7);
        Scalar att_err = qv.squaredNorm() + (1.0 - q0) * (1.0 - q0);

        Eigen::Vector3d f = u.template segment<3>(0);
        Eigen::Vector3d m = u.template segment<3>(3);

        return W_POS_STAGE * pos_err
             + W_VEL_STAGE * vel_err
             + W_ATT_STAGE * att_err
             + W_THRUST    * f.squaredNorm()
             + W_MOMENT    * m.squaredNorm();
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.template segment<3>(0) = 2.0 * W_POS_STAGE * (x.template segment<3>(0).eval() - target_pos_);
        g.template segment<3>(3) = 2.0 * W_VEL_STAGE * (x.template segment<3>(3).eval() - target_vel_);
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
// Terminal cost — runtime target
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
private:
    Eigen::Vector3d target_pos_;
    Eigen::Vector3d target_vel_;
public:
    TerminalCost(const Eigen::Vector3d& target_pos,
                 const Eigen::Vector3d& target_vel = Eigen::Vector3d::Zero())
        : target_pos_(target_pos), target_vel_(target_vel) {}

    Scalar p(const Vector<Scalar>& x) const override {
        Eigen::Vector3d pos = x.template segment<3>(0);
        Eigen::Vector3d vel = x.template segment<3>(3);
        return W_POS_TERM * (pos - target_pos_).squaredNorm()
             + W_VEL_TERM * (vel - target_vel_).squaredNorm();
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.template segment<3>(0) = 2.0 * W_POS_TERM * (x.template segment<3>(0).eval() - target_pos_);
        g.template segment<3>(3) = 2.0 * W_VEL_TERM * (x.template segment<3>(3).eval() - target_vel_);
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
// Max thrust constraint (unchanged)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class MaxThrustConstraint : public StageConstraintBase<Scalar> {
public:
    MaxThrustConstraint() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> c_n(1);
        c_n(0) = FMAX - u.template segment<3>(0).norm();
        return -c_n;
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
// Factory — target_state is now USED
//
// target_state layout:   [px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]
//
// Only position (0:2) and velocity (3:5) are used for the cost target.
// Attitude target is always identity (upright hover); angular rate target
// is always zero.  If target_state is empty or wrong size, fall back to
// (0, 0, 1) hover with zero velocity.
//
// create() is called ONCE (cold start or after setTerminalState()).
// On subsequent solves the caller updates x[0] and warm-starts U directly
// via setInitialState / setInitialControl — no need to call create() again.
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& target_state)
{
    // Extract target position and velocity from target_state.
    // Fall back to (0,0,1) / zero if target_state is not the expected size.
    Eigen::Vector3d target_pos(0.0, 0.0, 1.0);
    Eigen::Vector3d target_vel = Eigen::Vector3d::Zero();
    if (target_state.size() >= 6) {
        target_pos = target_state.segment<3>(0);
        target_vel = target_state.segment<3>(3);
    }

    auto prob = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    auto dyn = std::make_shared<Quad6DOF<double>>();
    dyn->setMass(MASS);
    dyn->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
    dyn->setJb(INERTIA);
    dyn->setDt(DT);
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageDynamics(i, dyn);

    // Pass the runtime target into every stage cost and the terminal cost.
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageCost(i, std::make_shared<StageCost<double>>(target_pos, target_vel));
    prob->setTerminalCost(std::make_shared<TerminalCost<double>>(target_pos, target_vel));

    auto mt = std::make_shared<MaxThrustConstraint<double>>();
    for (int i = 0; i < HORIZON; ++i)
        prob->addStageConstraint(i, mt);

    prob->setInitialState(0, current_state);

    // Cold-start U: gravity-cancelling thrust in body frame from current quaternion.
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