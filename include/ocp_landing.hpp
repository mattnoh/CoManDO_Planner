#pragma once

#include <Eigen/Dense>
#include <memory>
#include "optimal_control_problem.h"

namespace LandingOCP {

// ── Fixed parameters (mirroring standalone solve) ─────────────────────────────
const int    HORIZON = 100;                     // 10 s horizon (dt=0.05 → 5 s? Wait: 100*0.05=5s) 
                                                 // Standalone said "10 s horizon (dt=0.05)" but 100*0.05=5s. 
                                                 // Keep as given: HORIZON=100, DT=0.05 → 5s horizon.
const double DT      = 0.05;                     // seconds
const double MASS    = 0.027;                    // kg

// Scale inertia so that the diagonal entries become ~O(1)
const double J_SCALE = 1.0 / 1.66e-5;            // ≈ 60240
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() <<
    1.66e-5 * J_SCALE, 0.0, 0.0,
    0.0, 1.66e-5 * J_SCALE, 0.0,
    0.0, 0.0, 2.92e-5 * J_SCALE).finished();

// ── Constraint parameters (standalone values) ─────────────────────────────────
const double FMAX        = 1.2;                   // N (standalone used 1.2)
const double GLIDESLOPE  = 70.0;                  // degrees (unchanged)
const double THRUST_CONE = 60.0;                   // degrees (standalone used 60°)

// ─────────────────────────────────────────────────────────────────────────────
// Stage cost (mirroring standalone solve)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class StageCost : public StageCostBase<Scalar> {
private:
    Scalar thrust_weight_;
    Scalar moment_weight_;
    Scalar attitude_weight_;
    Scalar pos_weight_;

public:
    StageCost(Scalar thrust_weight = 1e-5,
              Scalar moment_weight = 1e-4,
              Scalar attitude_weight = 2.0,
              Scalar pos_weight = 0.0)
        : thrust_weight_(thrust_weight), moment_weight_(moment_weight),
          attitude_weight_(attitude_weight), pos_weight_(pos_weight) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Eigen::Vector3d f_B = u.template segment<3>(0);
        Eigen::Vector3d M_B = u.template segment<3>(3);

        // Attitude cost: penalize deviation from upright (q = [1,0,0,0])
        Scalar q0 = x(6);
        Eigen::Vector3d q_vec = x.template segment<3>(7);
        Scalar attitude_cost = attitude_weight_ * (q_vec.squaredNorm() + (1.0 - q0)*(1.0 - q0));

        // Running position cost on horizontal position only
        Scalar pos_cost = pos_weight_ * (x(0)*x(0) + x(1)*x(1));

        return thrust_weight_ * f_B.squaredNorm() + moment_weight_ * M_B.squaredNorm()
               + attitude_cost + pos_cost;
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Vector<Scalar> grad_x = Vector<Scalar>::Zero(x.size());

        // Attitude gradient
        Scalar q0 = x(6);
        Eigen::Vector3d q_vec = x.template segment<3>(7);
        grad_x(6) = -2.0 * attitude_weight_ * (1.0 - q0);
        grad_x.template segment<3>(7) = 2.0 * attitude_weight_ * q_vec;

        // Position gradient (horizontal only)
        grad_x(0) += 2.0 * pos_weight_ * x(0);
        grad_x(1) += 2.0 * pos_weight_ * x(1);

        return grad_x;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> grad_u = Vector<Scalar>::Zero(u.size());

        grad_u.template segment<3>(0) = 2.0 * thrust_weight_ * u.template segment<3>(0);
        grad_u.template segment<3>(3) = 2.0 * moment_weight_ * u.template segment<3>(3);

        return grad_u;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> H_x = Matrix<Scalar>::Zero(x.size(), x.size());

        // Attitude Hessian
        H_x(6, 6) = 2.0 * attitude_weight_;
        H_x.template block<3,3>(7, 7) = 2.0 * attitude_weight_ * Matrix<Scalar>::Identity(3, 3);

        // Position Hessian (horizontal only)
        H_x(0, 0) += 2.0 * pos_weight_;
        H_x(1, 1) += 2.0 * pos_weight_;

        return H_x;
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(u.size(), u.size());

        H.template block<3,3>(0, 0) = 2.0 * thrust_weight_ * Matrix<Scalar>::Identity(3, 3);
        H.template block<3,3>(3, 3) = 2.0 * moment_weight_ * Matrix<Scalar>::Identity(3, 3);

        return H;
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(x.size(), u.size());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Terminal cost (zero)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
public:
    Scalar p(const Vector<Scalar>& x) const override { (void)x; return 0.0; }
    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        return Vector<Scalar>::Zero(x.size());
    }
    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        return Matrix<Scalar>::Zero(x.size(), x.size());
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

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> c_n(1);
        c_n(0) = fmax_ - u.template segment<3>(0).norm();
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
// Thrust cone constraint (SOC)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class ThrustConeConstraint : public StageConstraintBase<Scalar> {
private:
    Scalar tan_tc_;
public:
    ThrustConeConstraint(Scalar theta_max_deg = THRUST_CONE) {
        tan_tc_ = std::tan(theta_max_deg * M_PI / 180.0);
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> c_n(3);
        c_n(0) = tan_tc_ * u(2);
        c_n(1) = u(0);
        c_n(2) = u(1);
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(3, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, u.size());
        J(0, 2) = tan_tc_;
        J(1, 0) = 1.0;
        J(2, 1) = 1.0;
        return -J;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Terminal equality constraint (pin all states to landed)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TerminalEqualityConstraint : public TerminalConstraintBase<Scalar> {
public:
    TerminalEqualityConstraint() {
        this->constraint_type = ConstraintType::EQ;
        this->dim_cT = 13;
    }

    Vector<Scalar> cT(const Vector<Scalar>& x) const override {
        Vector<Scalar> c(13);
        c(0)  = x(0)  - 0.0;
        c(1)  = x(1)  - 0.0;
        c(2)  = x(2)  - 0.0;
        c(3)  = x(3)  - 0.0;
        c(4)  = x(4)  - 0.0;
        c(5)  = x(5)  - 0.0;
        c(6)  = x(6)  - 1.0;
        c(7)  = x(7)  - 0.0;
        c(8)  = x(8)  - 0.0;
        c(9)  = x(9)  - 0.0;
        c(10) = x(10) - 0.0;
        c(11) = x(11) - 0.0;
        c(12) = x(12) - 0.0;
        return c;
    }

    Matrix<Scalar> cTx(const Vector<Scalar>& x) const override {
        (void)x;
        return Matrix<Scalar>::Identity(13, 13);
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

    // Stage cost: weights exactly as in standalone
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageCost(i, std::make_shared<StageCost<double>>(
            1e-5,    // thrust_weight
            1e-4,    // moment_weight
            2.0,     // attitude_weight
            0.0      // pos_weight (no running position cost)
        ));

    prob->setTerminalCost(std::make_shared<TerminalCost<double>>());

    // Terminal equality constraint
    prob->addTerminalConstraint(std::make_shared<TerminalEqualityConstraint<double>>());

    // Stage constraints with standalone parameters
    auto gs = std::make_shared<GlideslopeConstraint<double>>(GLIDESLOPE);
    auto tc = std::make_shared<ThrustConeConstraint<double>>(THRUST_CONE);
    auto mt = std::make_shared<MaxThrustConstraint<double>>(FMAX);
    for (int i = 0; i < HORIZON; ++i) {
        prob->addStageConstraint(i, gs);
        prob->addStageConstraint(i, tc);
        prob->addStageConstraint(i, mt);
    }

    // Initial state
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

} // namespace LandingOCP