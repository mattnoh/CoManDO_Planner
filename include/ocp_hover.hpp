#pragma once

#include <Eigen/Dense>
#include <memory>
#include "optimal_control_problem.h"

namespace HoverOCP {

// Fixed parameters for Hover OCP - use const instead of constexpr
const int HORIZON = 50;
const double DT = 0.05;  // 50 Hz
const double MASS = 0.027;
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() << 
    1.66e-5, 0.0, 0.0,
    0.0, 1.66e-5, 0.0,
    0.0, 0.0, 2.92e-5).finished();
const double CONTROL_WEIGHT = 1e-1;
const Eigen::Vector3d TARGET_POS = Eigen::Vector3d(0.0, 0.0, 1.0);

// Cost class
template <typename Scalar>
class Cost : public StageCostBase<Scalar> {
public:
    Cost() {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Eigen::Vector3d position = x.segment(0, 3);
        Eigen::Vector3d velocity = x.segment(3, 3);
        
        Scalar pos_error = (position - TARGET_POS).squaredNorm();
        Scalar vel_error = velocity.squaredNorm();
        
        Eigen::Vector3d f_B = u.segment(0, 3);
        Eigen::Vector3d M_B = u.segment(3, 3);
        Scalar thrust_cost = f_B.squaredNorm();
        Scalar moment_cost = M_B.squaredNorm();
        
        return 10.0 * pos_error + 1.0 * vel_error + 
               CONTROL_WEIGHT * thrust_cost + CONTROL_WEIGHT * moment_cost;
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u; // Unused parameter - suppress warning
        Vector<Scalar> grad = Vector<Scalar>::Zero(x.size());
        Eigen::Vector3d position = x.segment(0, 3);
        Eigen::Vector3d velocity = x.segment(3, 3);
        
        grad.segment(0, 3) = 20.0 * (position - TARGET_POS);
        grad.segment(3, 3) = 2.0 * velocity;
        return grad;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; // Unused parameter - suppress warning
        Vector<Scalar> grad = Vector<Scalar>::Zero(u.size());
        Eigen::Vector3d f_B = u.segment(0, 3);
        Eigen::Vector3d M_B = u.segment(3, 3);
        
        grad.segment(0, 3) = 2.0 * CONTROL_WEIGHT * f_B;
        grad.segment(3, 3) = 2.0 * CONTROL_WEIGHT * M_B;
        return grad;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused parameters - suppress warnings
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H.block(0, 0, 3, 3) = 20.0 * Matrix<Scalar>::Identity(3, 3);
        H.block(3, 3, 3, 3) = 2.0 * Matrix<Scalar>::Identity(3, 3);
        return H;
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused parameters - suppress warnings
        return 2.0 * CONTROL_WEIGHT * Matrix<Scalar>::Identity(u.size(), u.size());
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused parameters - suppress warnings
        return Matrix<Scalar>::Zero(x.size(), u.size());
    }
};

template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
public:
    TerminalCost() {}

    Scalar p(const Vector<Scalar>& x) const override {
        Eigen::Vector3d position = x.segment(0, 3);
        Eigen::Vector3d velocity = x.segment(3, 3);
        return 10.0 * (position - TARGET_POS).squaredNorm() + 1.0 * velocity.squaredNorm();
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> grad = Vector<Scalar>::Zero(x.size());
        Eigen::Vector3d position = x.segment(0, 3);
        Eigen::Vector3d velocity = x.segment(3, 3);
        grad.segment(0, 3) = 20.0 * (position - TARGET_POS);
        grad.segment(3, 3) = 2.0 * velocity;
        return grad;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        (void)x; // Unused parameter - suppress warning
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H.block(0, 0, 3, 3) = 20.0 * Matrix<Scalar>::Identity(3, 3);
        H.block(3, 3, 3, 3) = 2.0 * Matrix<Scalar>::Identity(3, 3);
        return H;
    }
};

// Create Hover OCP
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
        auto cost = std::make_shared<Cost<double>>();
        problem->setStageCost(i, cost);
    }
    
    // Setup terminal cost
    auto terminal_cost = std::make_shared<TerminalCost<double>>();
    problem->setTerminalCost(terminal_cost);
    
    // Set initial state
    problem->setInitialState(0, current_state);
    
    // Set initial control guess (hover)
    Eigen::Quaterniond q0(current_state(6), current_state(7), 
                         current_state(8), current_state(9));
    q0.normalize();
    Eigen::Vector3d f_B_hover = q0.inverse() * Eigen::Vector3d(0, 0, MASS * 9.81);
    Eigen::VectorXd u0(6);
    u0 << f_B_hover(0), f_B_hover(1), f_B_hover(2), 0.0, 0.0, 0.0;
    
    for (int i = 0; i < HORIZON; ++i) {
        problem->setInitialControl(i, u0);
    }
    
    return problem;
}

} // namespace HoverOCP