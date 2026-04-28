/// @file ocp_landing.hpp
/// @brief OCP formulation for landing problem (for online replanning).

#pragma once

#include <Eigen/Dense>
#include <memory>
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
#include "core/ocp_descriptor.hpp"

namespace LandingOCP {

// ── Fixed parameters ──────────────────────────────────────────────────────────
const int HORIZON = 100;
const double DT = 0.05;
const double MASS = 0.027;
const int DEFAULT_N_REPLAY = 4;
const double DEFAULT_MASS_KG = MASS;

const double J_SCALE = 1.0 / 1.66e-5;
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() <<
 1.66e-5 * J_SCALE, 0.0, 0.0,
 0.0, 1.66e-5 * J_SCALE, 0.0,
 0.0, 0.0, 2.92e-5 * J_SCALE).finished();

// ── Constraint parameters ─────────────────────────────────────────────────────
const double FMIN = 0.08;
const double FMAX = 0.6;
const double GLIDESLOPE = 60.0;
const double TILT_CONE = 60.0;

const double L_ARM = 0.046;
const double F_MOTOR_MAX = FMAX / 4.0;
const double F_MOTOR_MIN = FMIN / 4.0;
const double C_TAU = 0.005;
const double TAU_XY_MAX = L_ARM * (F_MOTOR_MAX - F_MOTOR_MIN);
const double TAU_Z_MAX = C_TAU * (F_MOTOR_MAX - F_MOTOR_MIN) * 4.0;
const double TAU_MAX_SCALED = TAU_XY_MAX * J_SCALE;

const double J_MAX = 0.05 / DT;

// ── Trajectory-consistency penalty weights ────────────────────────────────────
const double W_TRAJ_POS = 0.0;
const double W_TRAJ_VEL = 0.0;
const double W_TRAJ_TAPER_FACTOR = 0.1;
const int TAPER_NODES = 40;

// ── Solver parameters ─────────────────────────────────────────────────────────
const double SOLVER_REG1_MIN = 1e-6;
const double SOLVER_REG2_MIN = 1e-2;
const double SOLVER_MU_MUL = 0.1;
const double SOLVER_RHO = 10.0;
const double SOLVER_RHO_MUL = 10.0;
const double SOLVER_TOLERANCE = 1e-6;
const int SOLVER_MAX_ITER = 500;
const double SOLVER_RHOT = 1.0;

// ── Q: running state cost ─────────────────────────────────────────────────────
static const Eigen::VectorXd Q_DIAG = (Eigen::VectorXd(13) <<
 2.0, 2.0, 2.0,
 3.0, 3.0, 4.0,
 0.1,
 0.1, 0.1, 0.1,
 0.05, 0.05, 0.05).finished();

// ── R: running control cost ───────────────────────────────────────────────────
static const Eigen::VectorXd R_DIAG = (Eigen::VectorXd(4) <<
 1e-3,
 1e-4 / (J_SCALE*J_SCALE),
 1e-4 / (J_SCALE*J_SCALE),
 1e-4 / (J_SCALE*J_SCALE)).finished();

// ── S: delta-u slew-rate penalty ─────────────────────────────────────────────
static const Eigen::VectorXd S_DIAG = (Eigen::VectorXd(4) <<
 5e-3,
 1e-1 / (J_SCALE*J_SCALE),
 1e-1 / (J_SCALE*J_SCALE),
 1e-1 / (J_SCALE*J_SCALE)).finished();

// ── P: terminal state cost ────────────────────────────────────────────────────
static const Eigen::VectorXd P_DIAG = (Eigen::VectorXd(13) <<
 1000.0, 1000.0, 1000.0,
 500.0, 500.0, 500.0,
 500.0,
 500.0, 500.0, 500.0,
 200.0, 200.0, 200.0).finished();

// ── Solver params getter ──────────────────────────────────────────────────────
inline Param getSolverParams() {
 Param p;
 p.reg1_min = SOLVER_REG1_MIN;
 p.reg2_min = SOLVER_REG2_MIN;
 p.mu_mul = SOLVER_MU_MUL;
 p.rho = SOLVER_RHO;
 p.rhoT = SOLVER_RHOT;
 p.rho_mul = SOLVER_RHO_MUL;
 p.tolerance = SOLVER_TOLERANCE;
 p.max_iter = SOLVER_MAX_ITER;
 p.is_quaternion_in_state = false;
 return p;
}

// ── References ────────────────────────────────────────────────────────────────
static Eigen::VectorXd make_x_ref() {
 Eigen::VectorXd xr = Eigen::VectorXd::Zero(13);
 xr(2) = 0.1;
 xr(6) = 1.0;
 return xr;
}
static Eigen::VectorXd make_u_ref() {
 Eigen::VectorXd ur = Eigen::VectorXd::Zero(4);
 ur(0) = MASS * 9.81;
 return ur;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage cost with trajectory-consistency penalty
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TrajectoryAwareStageCost : public StageCostBase<Scalar> {
 Eigen::VectorXd x_land_ref_, u_ref_;
 Eigen::VectorXd Q_diag_, R_diag_, S_diag_, u_prev_;
 Eigen::VectorXd x_traj_ref_;
 double w_pos_;
 double w_vel_;
 bool has_traj_ref_;

public:
 TrajectoryAwareStageCost(const Eigen::VectorXd& x_land_ref,
    const Eigen::VectorXd& u_ref,
    const Eigen::VectorXd& Q_diag,
    const Eigen::VectorXd& R_diag,
    const Eigen::VectorXd& S_diag,
    const Eigen::VectorXd& u_prev,
    const Eigen::VectorXd& x_traj_ref,
    double w_pos, double w_vel)
 : x_land_ref_(x_land_ref), u_ref_(u_ref),
 Q_diag_(Q_diag), R_diag_(R_diag), S_diag_(S_diag), u_prev_(u_prev),
 x_traj_ref_(x_traj_ref), w_pos_(w_pos), w_vel_(w_vel),
 has_traj_ref_(x_traj_ref.size() >= 6)
 {}

 Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 Vector<Scalar> ex = x - x_land_ref_;
 Vector<Scalar> eu = u - u_ref_;
 Vector<Scalar> du = u - u_prev_;
 Scalar cost = ex.dot(Q_diag_.asDiagonal() * ex)
 + eu.dot(R_diag_.asDiagonal() * eu)
 + du.dot(S_diag_.asDiagonal() * du);
 if (has_traj_ref_) {
 Vector<Scalar> dp = x.head(3) - x_traj_ref_.head(3).template cast<Scalar>();
 Vector<Scalar> dv = x.segment(3,3) - x_traj_ref_.segment(3,3).template cast<Scalar>();
 cost += static_cast<Scalar>(w_pos_) * dp.squaredNorm()
 + static_cast<Scalar>(w_vel_) * dv.squaredNorm();
 }
 return cost;
 }

 Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)u;
 Vector<Scalar> grad = 2.0 * (Q_diag_.asDiagonal() * (x - x_land_ref_));
 if (has_traj_ref_) {
 grad.head(3) += 2.0 * static_cast<Scalar>(w_pos_)
 * (x.head(3) - x_traj_ref_.head(3).template cast<Scalar>());
 grad.segment(3,3) += 2.0 * static_cast<Scalar>(w_vel_)
 * (x.segment(3,3) - x_traj_ref_.segment(3,3).template cast<Scalar>());
 }
 return grad;
 }

 Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x;
 return 2.0 * (R_diag_.asDiagonal() * (u - u_ref_))
 + 2.0 * (S_diag_.asDiagonal() * (u - u_prev_));
 }

 Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 Eigen::VectorXd q2 = 2.0 * Q_diag_;
 if (has_traj_ref_) {
 q2(0) += 2.0 * w_pos_; q2(1) += 2.0 * w_pos_; q2(2) += 2.0 * w_pos_;
 q2(3) += 2.0 * w_vel_; q2(4) += 2.0 * w_vel_; q2(5) += 2.0 * w_vel_;
 }
 return q2.asDiagonal();
 }

 Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 return (2.0 * (R_diag_ + S_diag_)).asDiagonal();
 }

 Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 return Matrix<Scalar>::Zero(x.size(), u.size());
 }
};

// ─────────────────────────────────────────────────────────────────────────────
// Terminal cost
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class GenericTerminalCost : public TerminalCostBase<Scalar> {
 Eigen::VectorXd x_ref_, P_diag_;
public:
 GenericTerminalCost(const Eigen::VectorXd& x_ref, const Eigen::VectorXd& P_diag)
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
// Constraints
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class MaxThrustConstraint : public StageConstraintBase<Scalar> {
 Scalar fmax_;
public:
 explicit MaxThrustConstraint(Scalar fmax = static_cast<Scalar>(FMAX)) : fmax_(fmax) {
 this->constraint_type = ConstraintType::NO;
 this->dim_c = 1;
 }
 Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; Vector<Scalar> c_n(1); c_n(0) = u(0) - fmax_; return c_n;
 }
 Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u; return Matrix<Scalar>::Zero(1, x.size());
 }
 Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size()); J(0,0) = 1.0; return J;
 }
};

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

template <typename Scalar>
class GlideslopeConstraint : public StageConstraintBase<Scalar> {
 Scalar tan_gs_;
public:
 explicit GlideslopeConstraint(Scalar glideslope_deg = static_cast<Scalar>(GLIDESLOPE)) {
 tan_gs_ = std::tan(glideslope_deg * M_PI / 180.0);
 this->constraint_type = ConstraintType::SOC;
 this->dim_c = 3;
 }
 Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)u; Vector<Scalar> c_n(3);
 c_n(0) = tan_gs_ * x(2); c_n(1) = x(0); c_n(2) = x(1); return -c_n;
 }
 Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 Matrix<Scalar> J = Matrix<Scalar>::Zero(3, x.size());
 J(0,2) = tan_gs_; J(1,0) = 1.0; J(2,1) = 1.0; return -J;
 }
 Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u; return Matrix<Scalar>::Zero(3, u.size());
 }
};

template <typename Scalar>
class TiltConeConstraint : public StageConstraintBase<Scalar> {
 Scalar tilt_limit_;
public:
 explicit TiltConeConstraint(Scalar theta_max_deg = static_cast<Scalar>(TILT_CONE)) {
 tilt_limit_ = std::sqrt((1.0 - std::cos(theta_max_deg * M_PI / 180.0)) / 2.0);
 this->constraint_type = ConstraintType::SOC;
 this->dim_c = 3;
 }
 Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)u; Vector<Scalar> c_n(3);
 c_n(0) = tilt_limit_; c_n(1) = x(7); c_n(2) = x(8); return -c_n;
 }
 Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 Matrix<Scalar> J = Matrix<Scalar>::Zero(3, x.size());
 J(1,7) = 1.0; J(2,8) = 1.0; return -J;
 }
 Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u; return Matrix<Scalar>::Zero(3, u.size());
 }
};

template <typename Scalar>
class MaxMomentConstraint : public StageConstraintBase<Scalar> {
 Scalar tau_max_;
public:
 explicit MaxMomentConstraint(Scalar tau_max = static_cast<Scalar>(TAU_MAX_SCALED))
 : tau_max_(tau_max) {
 this->constraint_type = ConstraintType::SOC;
 this->dim_c = 4;
 }
 Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; Vector<Scalar> c_n(4);
 c_n(0) = tau_max_; c_n(1) = u(1); c_n(2) = u(2); c_n(3) = u(3); return -c_n;
 }
 Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u; return Matrix<Scalar>::Zero(4, x.size());
 }
 Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 (void)x; (void)u;
 Matrix<Scalar> J = Matrix<Scalar>::Zero(4, u.size());
 J(1,1) = 1.0; J(2,2) = 1.0; J(3,3) = 1.0; return -J;
 }
};

template <typename Scalar>
class VelocityJerkConstraint : public StageConstraintBase<Scalar> {
 double mass_, dt_, dv_max_;
 Eigen::Vector3d gravity_;
public:
 VelocityJerkConstraint(double mass, double dt,
    const Eigen::Vector3d& gravity, double j_max)
 : mass_(mass), dt_(dt), gravity_(gravity), dv_max_(j_max * dt) {
 this->constraint_type = ConstraintType::NO;
 this->dim_c = 6;
 }
 Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 const Eigen::Vector4d q = x.segment(6, 4);
 const double fz = u(0);
 const Eigen::Matrix3d C = Quad6DOF<Scalar>::calcC(q);
 const Eigen::Vector3d dv = dt_ * (C * Eigen::Vector3d(0.0, 0.0, fz) / mass_ + gravity_);
 Vector<Scalar> c_n(6);
 for (int i = 0; i < 3; ++i) {
 c_n(i) = dv(i) - dv_max_;
 c_n(3 + i) = -dv(i) - dv_max_;
 }
 return c_n;
 }
 Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 const Eigen::Vector4d q = x.segment(6, 4);
 const double fz = u(0);
 const Eigen::Vector3d f_B(0.0, 0.0, fz);
 Eigen::Matrix3d dCdq0, dCdq1, dCdq2, dCdq3;
 dCdq0 << 0,-2*q(3),2*q(2), 2*q(3),0,-2*q(1), -2*q(2),2*q(1),0;
 dCdq1 << 0,2*q(2),2*q(3), 2*q(2),-4*q(1),-2*q(0), 2*q(3),2*q(0),-4*q(1);
 dCdq2 << -4*q(2),2*q(1),2*q(0), 2*q(1),0,2*q(3), -2*q(0),2*q(3),-4*q(2);
 dCdq3 << -4*q(3),-2*q(0),2*q(1), 2*q(0),-4*q(3),2*q(2), 2*q(1),2*q(2),0;
 Eigen::Matrix<double,3,4> ddv_dq;
 ddv_dq.col(0) = (dt_/mass_)*(dCdq0*f_B);
 ddv_dq.col(1) = (dt_/mass_)*(dCdq1*f_B);
 ddv_dq.col(2) = (dt_/mass_)*(dCdq2*f_B);
 ddv_dq.col(3) = (dt_/mass_)*(dCdq3*f_B);
 Matrix<Scalar> J = Matrix<Scalar>::Zero(6, x.size());
 J.block(0,6,3,4) = ddv_dq;
 J.block(3,6,3,4) = -ddv_dq;
 return J;
 }
 Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 const Eigen::Vector4d q = x.segment(6, 4);
 const Eigen::Matrix3d C = Quad6DOF<Scalar>::calcC(q);
 const Eigen::Vector3d ddv_dfz = (dt_/mass_) * C.col(2);
 Matrix<Scalar> J = Matrix<Scalar>::Zero(6, u.size());
 J.block(0,0,3,1) = ddv_dfz;
 J.block(3,0,3,1) = -ddv_dfz;
 return J;
 }
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
 const Eigen::VectorXd& current_state,
 const Eigen::VectorXd& /* terminal_state */,
 const std::vector<Eigen::VectorXd>& prev_U = {},
 const std::vector<Eigen::VectorXd>& prev_X = {})
{
 auto prob = std::make_shared<OptimalControlProblem<double>>(HORIZON);

 auto dyn = std::make_shared<Quad6DOF<double>>();
 dyn->setMass(MASS);
 dyn->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
 dyn->setJb(INERTIA);
 dyn->setDt(DT);
 for (int i = 0; i < HORIZON; ++i)
 prob->setStageDynamics(i, dyn);

 const Eigen::VectorXd x_land_ref = make_x_ref();
 const Eigen::VectorXd u_ref = make_u_ref();
 const Eigen::VectorXd empty_traj_ref;

 for (int i = 0; i < HORIZON; ++i) {
 const Eigen::VectorXd& u_prev =
 (prev_U.size() > static_cast<size_t>(i)) ? prev_U[i] : u_ref;

 const Eigen::VectorXd& x_traj_ref =
 (prev_X.size() > static_cast<size_t>(i + 1))
 ? prev_X[i + 1]
 : empty_traj_ref;

 const bool in_taper = (i >= HORIZON - TAPER_NODES);
 const double w_pos = in_taper ? W_TRAJ_POS * W_TRAJ_TAPER_FACTOR : W_TRAJ_POS;
 const double w_vel = in_taper ? W_TRAJ_VEL * W_TRAJ_TAPER_FACTOR : W_TRAJ_VEL;

 prob->setStageCost(i, std::make_shared<TrajectoryAwareStageCost<double>>(
 x_land_ref, u_ref, Q_DIAG, R_DIAG, S_DIAG, u_prev,
 x_traj_ref, w_pos, w_vel));
 }

 prob->setTerminalCost(std::make_shared<GenericTerminalCost<double>>(x_land_ref, P_DIAG));

 auto gs = std::make_shared<GlideslopeConstraint<double>>(GLIDESLOPE);
 auto tc = std::make_shared<TiltConeConstraint<double>>(TILT_CONE);
 auto mt = std::make_shared<MaxThrustConstraint<double>>(FMAX);
 auto fmin = std::make_shared<MinThrustConstraint<double>>();
 auto mm = std::make_shared<MaxMomentConstraint<double>>();
 auto jerk = std::make_shared<VelocityJerkConstraint<double>>(
 MASS, DT, Eigen::Vector3d(0.0, 0.0, -9.81), J_MAX);

 for (int i = 0; i < HORIZON; ++i) {
 prob->addStageConstraint(i, tc);
 prob->addStageConstraint(i, mt);
 prob->addStageConstraint(i, fmin);
 }

 prob->setInitialState(0, current_state);

 const Eigen::VectorXd u_hover = make_u_ref();
 for (int i = 0; i < HORIZON; ++i) {
 const Eigen::VectorXd& u_ctrl =
 (prev_U.size() > static_cast<size_t>(i)) ? prev_U[i] : u_hover;
 prob->setInitialControl(i, u_ctrl);
 }

 return prob;
}

inline OCPDescriptor descriptor() {
    OCPDescriptor d;
    d.name = "landing";
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
        return create(a.current_state, a.terminal_state, a.prev_U, a.prev_X);
    };
    return d;
}

} // namespace LandingOCP
