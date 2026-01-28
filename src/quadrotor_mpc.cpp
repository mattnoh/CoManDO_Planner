#include "quadrotor_mpc.hpp"
#include <iostream>
#include <iomanip>

using namespace std;

// ========== CORRECT GOAL TRACKING COST (FROM ORIGINAL) ==========

template <typename Scalar>
class GoalTrackingCost : public StageCostBase<Scalar> {
private:
    Eigen::Vector3d goal_position_;
    Scalar control_weight_;
    bool is_terminal_stage_;

public:
    GoalTrackingCost(const Eigen::Vector3d& goal_pos, 
                     Scalar control_weight = 1e-3,
                     bool is_terminal = false)
        : goal_position_(goal_pos), 
          control_weight_(control_weight),
          is_terminal_stage_(is_terminal) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        // State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        Eigen::Vector3d position = x.segment(0, 3);
        Eigen::Vector3d velocity = x.segment(3, 3);
        
        // NO special terminal stage weighting - constant regulation
        Scalar pos_weight = 10.0;
        Scalar pos_error = (position - goal_position_).squaredNorm();
        
        // Velocity penalty (want zero velocity at goal)
        Scalar vel_weight = 1.0;
        Scalar vel_error = velocity.squaredNorm();
        
        // Hover thrust reference (mass * g in body frame, assuming upright)
        Eigen::Vector3d f_hover(0.0, 0.0, 0.027 * 9.81);
        Eigen::Vector3d f_B = u.segment(0, 3);
        Scalar thrust_deviation = (f_B - f_hover).squaredNorm();
        
        // Moment penalty (want zero moments)
        Eigen::Vector3d M_B = u.segment(3, 3);
        Scalar moment_cost = control_weight_ * M_B.squaredNorm();
        
        return pos_weight * pos_error + vel_weight * vel_error + 
               control_weight_ * thrust_deviation + moment_cost;
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u; // Unused
        Vector<Scalar> grad = Vector<Scalar>::Zero(x.size());
        
        Eigen::Vector3d position = x.segment(0, 3);
        Eigen::Vector3d velocity = x.segment(3, 3);
        
        Scalar pos_weight = is_terminal_stage_ ? 100.0 : 10.0;
        Scalar vel_weight = is_terminal_stage_ ? 10.0 : 1.0;
        
        // Gradient w.r.t. position
        grad.segment(0, 3) = 2.0 * pos_weight * (position - goal_position_);
        
        // Gradient w.r.t. velocity
        grad.segment(3, 3) = 2.0 * vel_weight * velocity;
        
        return grad;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; // Unused
        Vector<Scalar> grad = Vector<Scalar>::Zero(u.size());
        
        // Hover thrust reference
        Eigen::Vector3d f_hover(0.0, 0.0, 0.027 * 9.81);
        Eigen::Vector3d f_B = u.segment(0, 3);
        grad.segment(0, 3) = 2.0 * control_weight_ * (f_B - f_hover);
        
        // Moment gradient
        Eigen::Vector3d M_B = u.segment(3, 3);
        grad.segment(3, 3) = 2.0 * control_weight_ * M_B;
        
        return grad;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        
        Scalar pos_weight = is_terminal_stage_ ? 100.0 : 10.0;
        Scalar vel_weight = is_terminal_stage_ ? 10.0 : 1.0;
        
        // Hessian w.r.t. position
        H.block(0, 0, 3, 3) = 2.0 * pos_weight * Matrix<Scalar>::Identity(3, 3);
        
        // Hessian w.r.t. velocity
        H.block(3, 3, 3, 3) = 2.0 * vel_weight * Matrix<Scalar>::Identity(3, 3);
        
        return H;
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; // Unused
        return 2.0 * control_weight_ * Matrix<Scalar>::Identity(u.size(), u.size());
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused
        return Matrix<Scalar>::Zero(x.size(), u.size());
    }
};

template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
private:
    Eigen::Vector3d goal_position_;

public:
    TerminalCost(const Eigen::Vector3d& goal_pos) 
        : goal_position_(goal_pos) {}

    Scalar p(const Vector<Scalar>& x) const override {
        Eigen::Vector3d position = x.segment(0, 3);
        Eigen::Vector3d velocity = x.segment(3, 3);
        
        // SAME weights as stage cost - no special terminal behavior
        Scalar pos_weight = 10.0;
        Scalar vel_weight = 1.0;
        
        return pos_weight * (position - goal_position_).squaredNorm() + 
               vel_weight * velocity.squaredNorm();
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> grad = Vector<Scalar>::Zero(x.size());
        
        Eigen::Vector3d position = x.segment(0, 3);
        Eigen::Vector3d velocity = x.segment(3, 3);
        
        grad.segment(0, 3) = 2.0 * 10.0 * (position - goal_position_);
        grad.segment(3, 3) = 2.0 * 1.0 * velocity;
        
        return grad;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        
        H.block(0, 0, 3, 3) = 2.0 * 10.0 * Matrix<Scalar>::Identity(3, 3);
        H.block(3, 3, 3, 3) = 2.0 * 1.0 * Matrix<Scalar>::Identity(3, 3);
        
        return H;
    }
};

// ========== THRUST CONSTRAINT (for Quad6DOF - 6D control) ==========

template <typename Scalar>
class MaxThrustConstraint : public StageConstraintBase<Scalar> {
private:
    Scalar fmax_;

public:
    MaxThrustConstraint(Scalar fmax) : fmax_(fmax) {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; // Unused
        Vector<Scalar> c_n(1);
        Eigen::Vector3d f_B = u.segment(0, 3);
        c_n(0) = fmax_ - f_B.norm();
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; // Unused
        return Matrix<Scalar>::Zero(1, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; // Unused
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        Eigen::Vector3d f_B = u.segment(0, 3);
        Scalar norm_f = f_B.norm();
        
        if (norm_f > 1e-8) {
            J.block(0, 0, 1, 3) = -f_B.transpose() / norm_f;
        }
        
        return -J;
    }
};
// ========== MPC IMPLEMENTATION ==========

QuadrotorMPC::QuadrotorMPC(const Config& config) : config_(config) {
    // Setup ALIPDDP parameters
    solver_params_.reg1_min = 1e-6;
    solver_params_.reg2_min = 1.0;
    solver_params_.mu_mul = 0.1;
    solver_params_.rho = 20.0;
    solver_params_.rho_mul = 9.0;
    solver_params_.max_iter = 10;
    solver_params_.tolerance = 1e-3;
}

QuadrotorMPC::~QuadrotorMPC() = default;

void QuadrotorMPC::setupProblem(const Eigen::VectorXd& current_state) {
    // Always create fresh problem
    int N = config_.horizon;
    
    try {
        // Create problem
        problem_ = std::make_shared<OptimalControlProblem<double>>(N);
        auto dynamics = std::make_shared<Quad6DOF<double>>();
        dynamics->setMass(config_.mass);
        dynamics->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
        dynamics->setJb(config_.inertia);
        dynamics->setDt(config_.dt);
        
        // Set dynamics for all stages
        problem_->setStageDynamics(dynamics);
        
        // Extract goal position from terminal state
        Eigen::Vector3d goal_pos = config_.terminal_state.segment(0, 3);
        
        // Set stage costs
        for (int i = 0; i < N; ++i) {
            bool is_terminal_stage = (i == N - 1);
            auto cost = std::make_shared<GoalTrackingCost<double>>(
                goal_pos, 1e-3, is_terminal_stage);
            problem_->setStageCost(i, cost);
        }
        
        // Set terminal cost
        auto terminal_cost = std::make_shared<TerminalCost<double>>(goal_pos);
        problem_->setTerminalCost(terminal_cost);
        
        // Add thrust constraint
        auto thrust_constraint = std::make_shared<MaxThrustConstraint<double>>(config_.max_thrust);
        problem_->addStageConstraint(thrust_constraint);
    
        // Set initial state
        problem_->setInitialState(0, current_state);
        
        // Create initial guess for controls (hover)
        Eigen::Quaterniond q0(current_state(6), current_state(7), 
                             current_state(8), current_state(9));
        q0.normalize();
        Eigen::Vector3d f_B_hover = q0.inverse() * Eigen::Vector3d(0, 0, config_.mass * 9.81);
        Eigen::VectorXd u0(6);
        u0 << f_B_hover(0), f_B_hover(1), f_B_hover(2), 0.0, 0.0, 0.0;
        
        for (int i = 0; i < config_.horizon; ++i) {
            problem_->setInitialControl(i, u0);
        }
        
    } catch (const std::exception& e) {
        cerr << "ERROR in setupProblem: " << e.what() << endl;
        throw;
    }
}

QuadrotorMPC::Result QuadrotorMPC::solve(const Eigen::VectorXd& current_state) {
    Result result;
    result.success = false;
    auto start_time = chrono::high_resolution_clock::now();
    
    try {
        // Create fresh problem with current state
        setupProblem(current_state);
        
        if (!problem_) {
            cerr << "ERROR: Problem not initialized" << endl;
            return result;
        }
        
        // Create solver
        solver_.reset();
        solver_ = std::make_shared<ALIPDDP<double>>(*problem_);
        solver_->init(solver_params_);
        
        // Solve
        solver_->solve();
        
        // Get results
        std::vector<Eigen::VectorXd> X_result = solver_->getResX();
        auto U_result = solver_->getResU();
        
        auto end_time = chrono::high_resolution_clock::now();
        result.solve_time_ms = chrono::duration<double, milli>(end_time - start_time).count();
        
        if (X_result.size() > 1) {
            result.next_state = X_result[1];
            result.state_trajectory = X_result;
            result.control_trajectory = U_result;
            result.success = true;
            
            // Print first 4 states
            int states_to_print = std::min(4, (int)X_result.size());
            for (int i = 0; i < states_to_print; ++i) {
                cout << "X[" << i << "]: " << X_result[i].transpose() << endl;
            }
            
        } else {
            cerr << "ERROR: Empty trajectory from solver" << endl;
        }
        
    } catch (const std::exception& e) {
        cerr << "ERROR in solve: " << e.what() << endl;
    }
    
    return result;
}

void QuadrotorMPC::setTerminalState(const Eigen::VectorXd& terminal) {
    if (terminal.size() == 13) {
        config_.terminal_state = terminal;
    }
}