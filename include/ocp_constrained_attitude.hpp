#pragma once

#include <Eigen/Dense>
#include <memory>
#include "optimal_control_problem.h"

namespace ConstrainedAttitudeOCP {

// ── Fixed parameters ─────────────────────────────────────────────────────────
const int    HORIZON = 60;      // Increased from 30 to give more time to satisfy constraints
const double DT      = 0.02;    // 50 Hz
const double MASS    = 0.027;
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() <<
    1.66e-5, 0.0, 0.0,
    0.0, 1.66e-5, 0.0,
    0.0, 0.0, 2.92e-5).finished();

// ── Cost weights ─────────────────────────────────────────────────────────────
// Add position/velocity costs to keep trajectory smooth.
// Without these, the solver can take wild paths that satisfy constraints
// but cause numerical issues.
const double W_POS      = 0.0;    // Keep trajectory smooth
const double W_VEL      = 0.0;    // Penalize high velocities
const double W_ATTITUDE = 10.0;   // Drive to upright
const double W_THRUST   = 1e-4;   // Minimal control effort cost
const double W_MOMENT   = 1e-4;

// ── Constraint parameters ────────────────────────────────────────────────────
const double GLIDE_ANGLE = 70.0 * M_PI / 180.0;  // 70 degrees
const double FMAX        = 0.6;                   // Max thrust magnitude

// Reference attitude (upright)
const Eigen::Quaterniond Q_REF(1.0, 0.0, 0.0, 0.0);

// ─────────────────────────────────────────────────────────────────────────────
// Stage cost - REVERTED: Only attitude + control, NO position/velocity
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class Cost : public StageCostBase<Scalar> {
public:
    Cost() : q_ref_(Q_REF) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto pos   = x.segment(0, 3).eval();
        auto vel   = x.segment(3, 3).eval();
        auto quat  = x.segment(6, 4).eval();
        auto f     = u.segment(0, 3).eval();
        auto m     = u.segment(3, 3).eval();

        // Position cost
        Scalar pos_cost = W_POS * pos.squaredNorm();
        
        // Velocity cost
        Scalar vel_cost = W_VEL * vel.squaredNorm();

        // Attitude cost: simple quadratic on q_vec and (1-q0)
        Scalar q0 = quat(0);
        Eigen::Vector3d q_vec = quat.segment(1, 3);
        Scalar att_cost = W_ATTITUDE * (q_vec.squaredNorm() + (1.0 - q0)*(1.0 - q0));

        return pos_cost + vel_cost + att_cost
             + W_THRUST * f.squaredNorm()
             + W_MOMENT * m.squaredNorm();
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        
        // Position gradient
        g.segment(0, 3) = 2.0 * W_POS * x.segment(0, 3);
        
        // Velocity gradient
        g.segment(3, 3) = 2.0 * W_VEL * x.segment(3, 3);
        
        // Attitude gradient (simple quadratic form)
        Scalar q0 = x(6);
        Eigen::Vector3d q_vec = x.segment(7, 3);
        g(6) = -2.0 * W_ATTITUDE * (1.0 - q0);
        g.segment(7, 3) = 2.0 * W_ATTITUDE * q_vec;
        
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
        H.block(0, 0, 3, 3) = 2.0 * W_POS * Matrix<Scalar>::Identity(3, 3);
        H.block(3, 3, 3, 3) = 2.0 * W_VEL * Matrix<Scalar>::Identity(3, 3);
        H(6, 6) = 2.0 * W_ATTITUDE;
        H.block(7, 7, 3, 3) = 2.0 * W_ATTITUDE * Matrix<Scalar>::Identity(3, 3);
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
    Eigen::Quaterniond q_ref_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Terminal cost - Simple quadratic like your working example
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
public:
    Scalar p(const Vector<Scalar>& x) const override {
        // Simple quadratic on all states toward zero
        // The terminal equality constraint will enforce exact convergence
        Scalar cost = 0.0;
        
        // Position: drive to origin
        cost += 100.0 * x.segment(0, 3).squaredNorm();
        
        // Velocity: drive to zero
        cost += 10.0 * x.segment(3, 3).squaredNorm();
        
        // Quaternion: keep q0 close to 1 (rest will be constrained to zero)
        cost += 100.0 * (1.0 - x(6)) * (1.0 - x(6));
        cost += 100.0 * x.segment(7, 3).squaredNorm();
        
        // Angular velocity: drive to zero
        cost += 10.0 * x.segment(10, 3).squaredNorm();
        
        return cost;
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.segment(0, 3) = 200.0 * x.segment(0, 3);
        g.segment(3, 3) = 20.0 * x.segment(3, 3);
        g(6) = -200.0 * (1.0 - x(6));
        g.segment(7, 3) = 200.0 * x.segment(7, 3);
        g.segment(10, 3) = 20.0 * x.segment(10, 3);
        return g;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H.block(0, 0, 3, 3) = 200.0 * Matrix<Scalar>::Identity(3, 3);
        H.block(3, 3, 3, 3) = 20.0 * Matrix<Scalar>::Identity(3, 3);
        H(6, 6) = 200.0;
        H.block(7, 7, 3, 3) = 200.0 * Matrix<Scalar>::Identity(3, 3);
        H.block(10, 10, 3, 3) = 20.0 * Matrix<Scalar>::Identity(3, 3);
        return H;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Terminal equality constraint - KEEP dim_cT = 13 (all states to zero)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TerminalEqualityConstraint : public TerminalConstraintBase<Scalar> {
public:
    TerminalEqualityConstraint() {
        this->constraint_type = ConstraintType::EQ;
        this->dim_cT = 13; 
    }

    Vector<Scalar> cT(const Vector<Scalar>& x) const override {
        Vector<Scalar> cT_n(13);
        
        // Target: position at origin with altitude 0m, zero velocity,
        // upright orientation (q = [1,0,0,0]), zero angular velocity
        cT_n(0) = x(0) - 0.0;      // r_I,x = 0
        cT_n(1) = x(1) - 0.0;      // r_I,y = 0
        cT_n(2) = x(2) - 0.0;      // r_I,z = 0
        cT_n(3) = x(3) - 0.0;      // v_I,x = 0
        cT_n(4) = x(4) - 0.0;      // v_I,y = 0
        cT_n(5) = x(5) - 0.0;      // v_I,z = 0
        cT_n(6) = x(6) - 1.0;      // q0 = 1 (upright)
        cT_n(7) = x(7) - 0.0;      // q1 = 0
        cT_n(8) = x(8) - 0.0;      // q2 = 0
        cT_n(9) = x(9) - 0.0;      // q3 = 0
        cT_n(10) = x(10) - 0.0;    // ω_B,x = 0
        cT_n(11) = x(11) - 0.0;    // ω_B,y = 0
        cT_n(12) = x(12) - 0.0;    // ω_B,z = 0
        
        return cT_n;
    }

    Matrix<Scalar> cTx(const Vector<Scalar>& x) const override {
        return Matrix<Scalar>::Identity(13, 13);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Glideslope SOC constraint (3D cone: ||[x,y]|| <= tan(angle)*z)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class GlideslopeConstraint : public StageConstraintBase<Scalar> {
public:
    GlideslopeConstraint() {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;                      // cone dimension
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Vector<Scalar> c(3);
        c(0) = std::tan(GLIDE_ANGLE) * x(2);  // radial bound
        c(1) = x(0);                          // x
        c(2) = x(1);                          // y
        return c;                             // constraint: norm(c[1:2]) <= c[0]
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, x.size());
        J(0, 2) = std::tan(GLIDE_ANGLE);
        J(1, 0) = 1.0;
        J(2, 1) = 1.0;
        return J;
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(3, u.size());
    }

};

// ─────────────────────────────────────────────────────────────────────────────
// Max thrust SOC constraint (4D cone: ||[f_x, f_y, f_z]|| <= FMAX)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class MaxThrustConstraint : public StageConstraintBase<Scalar> {
public:
    MaxThrustConstraint() {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 4;                      // cone dimension
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> c(4);
        auto f = u.segment(0, 3).eval();
        c(0) = FMAX;                         // radial bound
        c.segment(1, 3) = f;                // f_x, f_y, f_z
        return c;                            // constraint: norm(c[1:3]) <= c[0]
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(4, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(4, u.size());
        J.block(1, 0, 3, 3) = Matrix<Scalar>::Identity(3, 3);
        return J;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& current_state)
{
    auto prob = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    // Dynamics
    auto dyn = std::make_shared<Quad6DOF<double>>();
    dyn->setMass(MASS);
    dyn->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
    dyn->setJb(INERTIA);
    dyn->setDt(DT);
    // Set dynamics for each stage
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageDynamics(i, dyn);

    // Stage cost
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageCost(i, std::make_shared<Cost<double>>());

    // Terminal cost (returns 0)
    prob->setTerminalCost(std::make_shared<TerminalCost<double>>());

    // Terminal equality constraint (all states = 0)
    prob->addTerminalConstraint(std::make_shared<TerminalEqualityConstraint<double>>());

    // SOC constraints
    auto glideslope = std::make_shared<GlideslopeConstraint<double>>();
    auto max_thrust = std::make_shared<MaxThrustConstraint<double>>();
    
    // Add stage constraints for every time step
    for (int i = 0; i < HORIZON; ++i) {
        prob->addStageConstraint(i, glideslope);
        prob->addStageConstraint(i, max_thrust);
    }

    // Initial state
    prob->setInitialState(0, current_state);

    // Warm-start: gravity-canceling thrust (in body frame)
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

} // namespace ConstrainedAttitudeOCP