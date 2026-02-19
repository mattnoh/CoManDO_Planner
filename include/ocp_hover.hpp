#pragma once

#include <Eigen/Dense>
#include <memory>
#include "optimal_control_problem.h"

namespace HoverOCP {

// ── Fixed parameters ─────────────────────────────────────────────────────────
const int    HORIZON = 50;
const double DT      = 0.05;
const double MASS    = 0.027;
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() <<
    1.66e-5, 0.0, 0.0,
    0.0, 1.66e-5, 0.0,
    0.0, 0.0, 2.92e-5).finished();

// ── Stage cost weights ────────────────────────────────────────────────────────
// Keep these moderate — they guide the trajectory shape over the horizon.
const double W_POS_STAGE  = 10.0;
const double W_VEL_STAGE  = 1.0;
const double W_THRUST     = 1e-1;
const double W_MOMENT     = 1e-1;

// ── Terminal cost weights ─────────────────────────────────────────────────────
// These must be large enough that the solver commits to *actually arriving*
// at the target with low velocity, not just trending toward it.
// Rule of thumb: ~10–20× the stage weight times the horizon length.
const double W_POS_TERM   = 500.0;
const double W_VEL_TERM   = 50.0;

// ─────────────────────────────────────────────────────────────────────────────
// Stage cost
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class Cost : public StageCostBase<Scalar> {
public:
    explicit Cost(const Eigen::VectorXd& target) {
        tgt_pos_ = target.segment(0, 3);
        tgt_vel_ = target.segment(3, 3);
    }

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto pos = x.segment(0, 3).eval();
        auto vel = x.segment(3, 3).eval();
        auto f   = u.segment(0, 3).eval();
        auto m   = u.segment(3, 3).eval();
        return W_POS_STAGE * (pos - tgt_pos_).squaredNorm()
             + W_VEL_STAGE * (vel - tgt_vel_).squaredNorm()
             + W_THRUST    * f.squaredNorm()
             + W_MOMENT    * m.squaredNorm();
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.segment(0, 3) = 2.0 * W_POS_STAGE * (x.segment(0, 3).eval() - tgt_pos_);
        g.segment(3, 3) = 2.0 * W_VEL_STAGE * (x.segment(3, 3).eval() - tgt_vel_);
        return g;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> g = Vector<Scalar>::Zero(u.size());
        g.segment(0, 3) = 2.0 * W_THRUST * u.segment(0, 3);
        g.segment(3, 3) = 2.0 * W_MOMENT * u.segment(3, 3);
        return g;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H.block(0, 0, 3, 3) = 2.0 * W_POS_STAGE * Matrix<Scalar>::Identity(3, 3);
        H.block(3, 3, 3, 3) = 2.0 * W_VEL_STAGE * Matrix<Scalar>::Identity(3, 3);
        return H;
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(u.size(), u.size());
        H.block(0, 0, 3, 3) = 2.0 * W_THRUST * Matrix<Scalar>::Identity(3, 3);
        H.block(3, 3, 3, 3) = 2.0 * W_MOMENT * Matrix<Scalar>::Identity(3, 3);
        return H;
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(x.size(), u.size());
    }

private:
    Eigen::Vector3d tgt_pos_, tgt_vel_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Terminal cost — deliberately much heavier than stage cost
// This is the primary mechanism that forces the drone to arrive and stop.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
public:
    explicit TerminalCost(const Eigen::VectorXd& target) {
        tgt_pos_ = target.segment(0, 3);
        tgt_vel_ = target.segment(3, 3);
    }

    Scalar p(const Vector<Scalar>& x) const override {
        return W_POS_TERM * (x.segment(0, 3).eval() - tgt_pos_).squaredNorm()
             + W_VEL_TERM * (x.segment(3, 3).eval() - tgt_vel_).squaredNorm();
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.segment(0, 3) = 2.0 * W_POS_TERM * (x.segment(0, 3).eval() - tgt_pos_);
        g.segment(3, 3) = 2.0 * W_VEL_TERM * (x.segment(3, 3).eval() - tgt_vel_);
        return g;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        (void)x;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H.block(0, 0, 3, 3) = 2.0 * W_POS_TERM * Matrix<Scalar>::Identity(3, 3);
        H.block(3, 3, 3, 3) = 2.0 * W_VEL_TERM * Matrix<Scalar>::Identity(3, 3);
        return H;
    }

private:
    Eigen::Vector3d tgt_pos_, tgt_vel_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory
//   current_state : 13-element state vector at current time
//   target_state  : desired final state (only position [0:3] and velocity [3:6]
//                   are used; pass zero velocity for a stationary hover)
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& target_state)
{
    auto prob = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    auto dyn = std::make_shared<Quad6DOF<double>>();
    dyn->setMass(MASS);
    dyn->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
    dyn->setJb(INERTIA);
    dyn->setDt(DT);
    prob->setStageDynamics(dyn);

    for (int i = 0; i < HORIZON; ++i)
        prob->setStageCost(i, std::make_shared<Cost<double>>(target_state));
    prob->setTerminalCost(std::make_shared<TerminalCost<double>>(target_state));

    prob->setInitialState(0, current_state);

    // Warm-start: gravity-cancelling thrust in body frame
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