#pragma once

#include <Eigen/Dense>
#include <memory>
#include <cmath>
#include "optimal_control_problem.h"

namespace ConstrainedAttitudeOCP {

// Fixed parameters
const int HORIZON = 30;
const double DT = 0.02;
const double MASS = 0.027;
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() << 
    1.66e-5, 0.0, 0.0,
    0.0, 1.66e-5, 0.0,
    0.0, 0.0, 2.92e-5).finished();
const double ATTITUDE_WEIGHT = 2.0;
const double FMAX = 40.0;
const double THETA_MAX = 60.0;  // degrees
const double GLIDESLOPE_ANGLE = 70.0;  // degrees
const double THRUST_WEIGHT = 1e-5;
const double MOMENT_WEIGHT = 1e-4;

// Terminal state (origin, upright)
inline Eigen::VectorXd getTerminalState() {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(13);
    state(6) = 1.0;  // upright quaternion
    return state;
}

// Cost Functions
template <typename Scalar>
class Cost : public StageCostBase<Scalar> {
public:
    Cost() {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Eigen::Vector3d f_B = u.segment(0, 3);
        Eigen::Vector3d M_B = u.segment(3, 3);
        
        // Attitude cost to drive reorientation
        Scalar q0 = x(6);
        Eigen::Vector3d q_vec = x.segment(7, 3);
        Scalar attitude_cost = ATTITUDE_WEIGHT * (q_vec.squaredNorm() + (1.0 - q0)*(1.0 - q0));
        
        return THRUST_WEIGHT * f_B.squaredNorm() + MOMENT_WEIGHT * M_B.squaredNorm() + attitude_cost;
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u; // Unused parameter - suppress warning
        Vector<Scalar> grad = Vector<Scalar>::Zero(x.size());
        Scalar q0 = x(6);
        Eigen::Vector3d q_vec = x.segment(7, 3);
        
        grad(6) = -2.0 * ATTITUDE_WEIGHT * (1.0 - q0);
        grad.segment(7, 3) = 2.0 * ATTITUDE_WEIGHT * q_vec;
        return grad;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; // Unused parameter - suppress warning
        Vector<Scalar> grad = Vector<Scalar>::Zero(u.size());
        grad.segment(0, 3) = 2.0 * THRUST_WEIGHT * u.segment(0, 3);
        grad.segment(3, 3) = 2.0 * MOMENT_WEIGHT * u.segment(3, 3);
        return grad;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused parameters - suppress warnings
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H(6, 6) = 2.0 * ATTITUDE_WEIGHT;
        H.block(7, 7, 3, 3) = 2.0 * ATTITUDE_WEIGHT * Matrix<Scalar>::Identity(3, 3);
        return H;
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused parameters - suppress warnings
        Matrix<Scalar> H = Matrix<Scalar>::Zero(u.size(), u.size());
        H.block(0, 0, 3, 3) = 2.0 * THRUST_WEIGHT * Matrix<Scalar>::Identity(3, 3);
        H.block(3, 3, 3, 3) = 2.0 * MOMENT_WEIGHT * Matrix<Scalar>::Identity(3, 3);
        return H;
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused parameters - suppress warnings
        return Matrix<Scalar>::Zero(x.size(), u.size());
    }
};

template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
public:
    Scalar p(const Vector<Scalar>& x) const override { 
        (void)x; // Unused parameter - suppress warning
        return 0.0; 
    }
    
    Vector<Scalar> px(const Vector<Scalar>& x) const override { 
        (void)x; // Unused parameter - suppress warning
        return Vector<Scalar>::Zero(x.size()); 
    }
    
    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override { 
        (void)x; // Unused parameter - suppress warning
        return Matrix<Scalar>::Zero(x.size(), x.size()); 
    }
};

// Constraints
template <typename Scalar>
class MaxThrustConstraint : public StageConstraintBase<Scalar> {
public:
    MaxThrustConstraint() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; // Unused parameter - suppress warning
        Vector<Scalar> c_n(1);
        Eigen::Vector3d f_B = u.segment(0, 3);
        c_n(0) = FMAX - f_B.norm();
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused parameters - suppress warnings
        return Matrix<Scalar>::Zero(1, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; // Unused parameter - suppress warning
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        Eigen::Vector3d f_B = u.segment(0, 3);
        Scalar norm_f = f_B.norm();
        if (norm_f > 1e-8) {
            J.block(0, 0, 1, 3) = -f_B.transpose() / norm_f;
        }
        return -J;
    }
};

template <typename Scalar>
class GlideSlopeConeConstraint : public StageConstraintBase<Scalar> {
private:
    Scalar tan_glideslope;

public:
    GlideSlopeConeConstraint() {
        tan_glideslope = std::tan(GLIDESLOPE_ANGLE * M_PI / 180.0);
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u; // Unused parameter - suppress warning
        Vector<Scalar> c_n(3);
        Eigen::Vector3d r_I = x.segment(0, 3);
        c_n(0) = tan_glideslope * r_I(2);
        c_n(1) = r_I(0);
        c_n(2) = r_I(1);
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u; // Unused parameter - suppress warning
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, x.size());
        J(0, 2) = tan_glideslope;
        J(1, 0) = 1.0;
        J(2, 1) = 1.0;
        return -J;
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused parameters - suppress warnings
        return Matrix<Scalar>::Zero(3, u.size());
    }
};

template <typename Scalar>
class ThrustConeConstraint : public StageConstraintBase<Scalar> {
private:
    Scalar tan_theta_max;

public:
    ThrustConeConstraint() {
        tan_theta_max = std::tan(THETA_MAX * M_PI / 180.0);
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; // Unused parameter - suppress warning
        Vector<Scalar> c_n(3);
        Eigen::Vector3d f_B = u.segment(0, 3);
        c_n(0) = tan_theta_max * f_B(2);
        c_n(1) = f_B(0);
        c_n(2) = f_B(1);
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused parameters - suppress warnings
        return Matrix<Scalar>::Zero(3, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; // Unused parameter - suppress warning
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, u.size());
        J(0, 2) = tan_theta_max;
        J(1, 0) = 1.0;
        J(2, 1) = 1.0;
        return -J;
    }
};

template <typename Scalar>
class TerminalEqualityConstraint : public TerminalConstraintBase<Scalar> {
public:
    TerminalEqualityConstraint() {
        this->constraint_type = ConstraintType::EQ;
        this->dim_cT = 13;
    }

    Vector<Scalar> cT(const Vector<Scalar>& x) const override {
        Vector<Scalar> cT_n(13);
        Eigen::VectorXd terminal_state = getTerminalState();
        for (int i = 0; i < 13; ++i) {
            cT_n(i) = x(i) - terminal_state(i);
        }
        return cT_n;
    }

    Matrix<Scalar> cTx(const Vector<Scalar>& x) const override {
        (void)x; // Unused parameter - suppress warning
        return Matrix<Scalar>::Identity(13, 13);
    }
};

// Create Constrained Attitude OCP
inline std::shared_ptr<OptimalControlProblem<double>> create(const Eigen::VectorXd& current_state) {
    auto problem = std::make_shared<OptimalControlProblem<double>>(HORIZON);
    
    // Setup dynamics
    auto dynamics = std::make_shared<Quad6DOF<double>>();
    dynamics->setMass(MASS);
    dynamics->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
    dynamics->setJb(INERTIA);
    dynamics->setDt(DT);
    problem->setStageDynamics(dynamics);
    
    // Setup costs
    for (int i = 0; i < HORIZON; ++i) {
        problem->setStageCost(i, std::make_shared<Cost<double>>());
    }
    problem->setTerminalCost(std::make_shared<TerminalCost<double>>());
    
    // Setup constraints
    for (int i = 0; i < HORIZON; ++i) {
        problem->addStageConstraint(i, std::make_shared<MaxThrustConstraint<double>>());
        problem->addStageConstraint(i, std::make_shared<GlideSlopeConeConstraint<double>>());
        problem->addStageConstraint(i, std::make_shared<ThrustConeConstraint<double>>());
    }
    
    problem->addTerminalConstraint(std::make_shared<TerminalEqualityConstraint<double>>());
    
    // Set initial state
    problem->setInitialState(0, current_state);
    
    // Initial control guess
    Eigen::Quaterniond q0(current_state(6), current_state(7), 
                         current_state(8), current_state(9));
    q0.normalize();
    Eigen::Vector3d g_I(0.0, 0.0, -9.81);
    Eigen::Vector3d f_B_init = q0.inverse() * (-g_I * MASS);
    
    Eigen::VectorXd u0(6);
    u0 << f_B_init(0), f_B_init(1), f_B_init(2), 0.0, 0.0, 0.0;
    
    for (int i = 0; i < HORIZON; ++i) {
        problem->setInitialControl(i, u0);
    }
    
    return problem;
}

} // namespace ConstrainedAttitudeOCP