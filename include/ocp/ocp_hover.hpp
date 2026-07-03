// ocp_hover.hpp
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <cmath>
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
#include "planner_core/types.hpp"

namespace HoverOCP {

// ── Fixed parameters ─────────────────────────────────────────────
const int HORIZON = 100;
const double DT = 0.05;
const double MASS = 0.0282;
const int DEFAULT_N_REPLAY = 4;
const double DEFAULT_MASS_KG = MASS;

const double J_SCALE = 1.0 / 1.66e-5;
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() <<
 1.66e-5 * J_SCALE, 0.0, 0.0,
 0.0, 1.66e-5 * J_SCALE, 0.0,
 0.0, 0.0, 2.92e-5 * J_SCALE).finished();

// ── Constraint parameters ────────────────────────────────────────
// Keep the planned agility mild (thrust-to-weight ~1.5): the plan is tracked
// by the platform's own position controller (PX4 on this branch), and a
// cf-agility plan (FMAX 1.2, t/w 4.3) outruns it — tracking error then trips
// handoff recovery every replan and the drone limit-cycles around the target.
const double FMAX = 0.42;
const double FMIN = 0.08;

// ── Solver parameters ────────────────────────────────────────────
const double SOLVER_REG1_MIN = 1e-6;
const double SOLVER_REG2_MIN = 1.0;
const double SOLVER_MU_MUL = 0.1;
const double SOLVER_RHO = 20.0;
const double SOLVER_RHO_MUL = 9.0;
const double SOLVER_TOLERANCE = 1e-3;
const int SOLVER_MAX_ITER = 200;

// ── Stage cost weights ───────────────────────────────────────────
const double W_POS_STAGE = 15.0;
const double W_VEL_STAGE = 20.0;
const double W_ATT_STAGE = 2.0;
const double W_ANGRATE_STAGE = 5.0;
const double W_THRUST = 1e-1;
const double W_MOMENT = 1e-1;

// ── Terminal cost weights ────────────────────────────────────────
// High enough that the plan ends AT the target within the 5 s horizon even
// with the mild stage weights above — open_loop mode holds the plan's
// terminal state forever, so any terminal offset becomes a permanent offset.
const double TERM_POS_WEIGHT = 600.0;
const double TERM_VEL_WEIGHT = 100.0;
const double TERM_ATT_WEIGHT = 200.0;
const double TERM_ANGRATE_WEIGHT = 100.0;

// ─────────────────────────────────────────────────────────────────
inline Param getSolverParams() {
 Param p;
 p.reg1_min = SOLVER_REG1_MIN;
 p.reg2_min = SOLVER_REG2_MIN;
 p.mu_mul = SOLVER_MU_MUL;
 p.rho = SOLVER_RHO;
 p.rho_mul = SOLVER_RHO_MUL;
 p.tolerance = SOLVER_TOLERANCE;
 p.max_iter = SOLVER_MAX_ITER;
 p.is_quaternion_in_state = false;
 return p;
}

// ─────────────────────────────────────────────────────────────────
// Stage cost – penalises tracking error and control effort
// ─────────────────────────────────────────────────────────────────
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
 Eigen::Vector3d omega = x.template segment<3>(10);
 Scalar q0 = x(6);
 Eigen::Vector3d qv = x.template segment<3>(7);

 Scalar att_err = qv.squaredNorm() + (1.0 - q0) * (1.0 - q0);

 Scalar fz = u(0);
 Eigen::Vector3d m = u.template segment<3>(1);

 return W_POS_STAGE * (pos - target_pos_).squaredNorm()
 + W_VEL_STAGE * (vel - target_vel_).squaredNorm()
 + W_ATT_STAGE * att_err
 + W_ANGRATE_STAGE * omega.squaredNorm()
 + W_THRUST * fz * fz
 + W_MOMENT * m.squaredNorm();
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

 g.template segment<3>(10) = 2.0 * W_ANGRATE_STAGE * x.template segment<3>(10);

 return g;
 }

 Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x;
 Vector<Scalar> g = Vector<Scalar>::Zero(u.size());
 g(0) = 2.0 * W_THRUST * u(0);
 g.template segment<3>(1) = 2.0 * W_MOMENT * u.template segment<3>(1);
 return g;
 }

 Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());

 H.template block<3,3>(0,0) = 2.0 * W_POS_STAGE * Matrix<Scalar>::Identity(3,3);
 H.template block<3,3>(3,3) = 2.0 * W_VEL_STAGE * Matrix<Scalar>::Identity(3,3);
 H(6,6) = 2.0 * W_ATT_STAGE;
 H.template block<3,3>(7,7) = 2.0 * W_ATT_STAGE * Matrix<Scalar>::Identity(3,3);
 H.template block<3,3>(10,10) = 2.0 * W_ANGRATE_STAGE * Matrix<Scalar>::Identity(3,3);

 return H;
 }

 Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 Matrix<Scalar> H = Matrix<Scalar>::Zero(u.size(), u.size());
 H(0, 0) = 2.0 * W_THRUST;
 H.template block<3,3>(1,1) = 2.0 * W_MOMENT * Matrix<Scalar>::Identity(3,3);
 return H;
 }

 Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 return Matrix<Scalar>::Zero(x.size(), u.size());
 }
};

// ─────────────────────────────────────────────────────────────────
// Terminal cost – strong penalty on all states
// ─────────────────────────────────────────────────────────────────
template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
private:
 Eigen::Vector3d target_pos_;
 Eigen::Vector3d target_vel_;

public:
 TerminalCost(const Eigen::VectorXd& target_state)
 : target_pos_(target_state.segment<3>(0)),
  target_vel_(target_state.segment<3>(3))
 {}

 Scalar p(const Vector<Scalar>& x) const override {
 Eigen::Vector3d pos = x.template segment<3>(0);
 Eigen::Vector3d vel = x.template segment<3>(3);
 Scalar q0 = x(6);
 Eigen::Vector3d qv = x.template segment<3>(7);
 Eigen::Vector3d omega = x.template segment<3>(10);

 Scalar att_err = qv.squaredNorm() + (1.0 - q0) * (1.0 - q0);

 return TERM_POS_WEIGHT * (pos - target_pos_).squaredNorm()
 + TERM_VEL_WEIGHT * (vel - target_vel_).squaredNorm()
 + TERM_ATT_WEIGHT * att_err
 + TERM_ANGRATE_WEIGHT * omega.squaredNorm();
 }

 Vector<Scalar> px(const Vector<Scalar>& x) const override {
 Vector<Scalar> grad = Vector<Scalar>::Zero(x.size());

 grad.template segment<3>(0) = 2.0 * TERM_POS_WEIGHT * (x.template segment<3>(0) - target_pos_);
 grad.template segment<3>(3) = 2.0 * TERM_VEL_WEIGHT * (x.template segment<3>(3) - target_vel_);

 Scalar q0 = x(6);
 Eigen::Vector3d qv = x.template segment<3>(7);
 grad(6) = -2.0 * TERM_ATT_WEIGHT * (1.0 - q0);
 grad.template segment<3>(7) = 2.0 * TERM_ATT_WEIGHT * qv;

 grad.template segment<3>(10) = 2.0 * TERM_ANGRATE_WEIGHT * x.template segment<3>(10);

 return grad;
 }

 Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
 (void)x;
 Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());

 H.template block<3,3>(0,0) = 2.0 * TERM_POS_WEIGHT * Matrix<Scalar>::Identity(3,3);
 H.template block<3,3>(3,3) = 2.0 * TERM_VEL_WEIGHT * Matrix<Scalar>::Identity(3,3);
 H(6,6) = 2.0 * TERM_ATT_WEIGHT;
 H.template block<3,3>(7,7) = 2.0 * TERM_ATT_WEIGHT * Matrix<Scalar>::Identity(3,3);
 H.template block<3,3>(10,10) = 2.0 * TERM_ANGRATE_WEIGHT * Matrix<Scalar>::Identity(3,3);

 return H;
 }
};

// ─────────────────────────────────────────────────────────────────
// Max thrust constraint (inequality)
// ─────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────
// Min thrust constraint (inequality) – a quad cannot push downward
// ─────────────────────────────────────────────────────────────────
template <typename Scalar>
class MinThrustConstraint : public StageConstraintBase<Scalar> {
public:
 MinThrustConstraint() {
 this->constraint_type = ConstraintType::NO;
 this->dim_c = 1;
 }
 Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; Vector<Scalar> c_n(1); c_n(0) = FMIN - u(0); return c_n;
 }
 Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u; return Matrix<Scalar>::Zero(1, x.size());
 }
 Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size()); J(0,0) = -1.0; return J;
 }
};

// ─────────────────────────────────────────────────────────────────
// Factory function: creates a hover problem with given target
// ─────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
 const Eigen::VectorXd& current_state,
 const Eigen::VectorXd& terminal_state,
 double fmax = FMAX)
{
 auto prob = std::make_shared<OptimalControlProblem<double>>(HORIZON);

 auto dyn = std::make_shared<Quad6DOF<double>>();
 dyn->setMass(MASS);
 dyn->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
 dyn->setJb(INERTIA);
 dyn->setDt(DT);
 for (int i = 0; i < HORIZON; ++i)
 prob->setStageDynamics(i, dyn);

 Eigen::Vector3d target_pos = terminal_state.segment<3>(0);
 Eigen::Vector3d target_vel = terminal_state.segment<3>(3);
 auto stage_cost = std::make_shared<StageCost<double>>(target_pos, target_vel);
 for (int i = 0; i < HORIZON; ++i)
 prob->setStageCost(i, stage_cost);

 prob->setTerminalCost(std::make_shared<TerminalCost<double>>(terminal_state));

 auto mt = std::make_shared<MaxThrustConstraint<double>>(fmax);
 auto fmin = std::make_shared<MinThrustConstraint<double>>();
 for (int i = 0; i < HORIZON; ++i) {
 prob->addStageConstraint(i, mt);
 prob->addStageConstraint(i, fmin);
 }

 prob->setInitialState(0, current_state);

 Eigen::VectorXd u0(4);
 u0 << MASS * 9.81, 0.0, 0.0, 0.0;
 for (int i = 0; i < HORIZON; ++i)
 prob->setInitialControl(i, u0);

 return prob;
}

inline OCPDescriptor descriptor() {
    OCPDescriptor d;
    d.name = "hover";
    d.dt = DT;
    d.default_n_replay = DEFAULT_N_REPLAY;
    d.default_mass_kg = DEFAULT_MASS_KG;
    d.warm_start = OCPDescriptor::WarmStart::Shift;
    d.command_mode = OCPDescriptor::CommandMode::CmdFullState;
    d.drone_odom_mode = OCPDescriptor::DroneOdomMode::Absolute;
    d.state_dim = 13;
    d.control_dim = 4;
    d.log_state_headers = OCPLoggerDefaults::getStateHeaders13D();
    d.state_names = OCPLoggerDefaults::getStateNames13D();
    d.control_names = OCPLoggerDefaults::getControlNames4D();
    d.extract_actual_state_row = OCPLoggerDefaults::getActualStateRow13D;
    d.make_hover_state = OCPLoggerDefaults::makeHoverState13D;
    d.getSolverParams = getSolverParams;
    d.create = [](const OCPCreateArgs& a) {
        return create(a.current_state, a.terminal_state);
    };
    return d;
}

} // namespace HoverOCP
