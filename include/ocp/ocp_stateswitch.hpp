/// @file ocp_stateswitch.hpp
/// @brief OCP formulation for relative-frame landing on a moving target (state-switch).
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include "target/target_accel_buffer.hpp"
#include "target/circular_target.hpp"
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
#include "core/ocp_descriptor.hpp"

namespace StateswitchOCP {

// ── Dimensions ────────────────────────────────────────────────────────────────
static constexpr int NX = 13;
static constexpr int NU = 4;
static constexpr int NX_SS = 14;
static constexpr int NU_SS = 5;
static constexpr int IDX_DT = 13;
static constexpr int IDX_THETA = 4;

// ── Horizon / timing ─────────────────────────────────────────────────────────
static constexpr int HORIZON = 30;
static constexpr double TH_INIT = 0.1;
static constexpr double THL = 0.05;
static constexpr double THH = 0.2;

// ── Vehicle parameters ───────────────────────────────────────────────────────
static constexpr double MASS = 0.027;
static constexpr int DEFAULT_N_REPLAY = 7;
static constexpr double DEFAULT_MASS_KG = MASS;
static constexpr double IXX = 1.66e-5;
static constexpr double IYY = 1.66e-5;
static constexpr double IZZ = 2.92e-5;
static constexpr double J_SCALE = 1.0 / IXX;
static constexpr double FMIN = 0.08;
static constexpr double FMAX = 0.6;
static constexpr double L_ARM = 0.046;
static const double TAU_MAX = L_ARM * (FMAX/4.0 - FMIN/4.0) * J_SCALE;

static const Eigen::Matrix3d J_B = [](){
 Eigen::Matrix3d J; J.setZero();
 J(0,0)=IXX*J_SCALE; J(1,1)=IYY*J_SCALE; J(2,2)=IZZ*J_SCALE;
 return J;
}();
static const Eigen::Vector3d GRAVITY(0.0, 0.0, -9.81);

// ── Glideslope parameters ─────────────────────────────────────────────────────
static constexpr double GS_DEG = 60.0;
static const double GS_TAN = std::tan(GS_DEG * M_PI / 180.0);
static constexpr double VZ_LAND_MAX = 1.5;

// ── Solver parameters ─────────────────────────────────────────────────────────
const double SOLVER_REG1_MIN = 1e-2;
const double SOLVER_REG2_MIN = 0.5;
const double SOLVER_MU_MUL = 0.1;
const double SOLVER_RHO = 10.0;
const double SOLVER_RHOT = 1.0;
const double SOLVER_RHO_MUL = 10.0;
const double SOLVER_TOLERANCE = 1e-4;
const int SOLVER_MAX_ITER = 500;

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

struct StateswitchExtra {
 target_models::TargetAccelBuffer buf;
 double t0_abs = 0.0;
};

// ── Quad6DOFVarTimeRelativeTV wrapper ───────────────────────────────────────
// Time-varying target acceleration variant. Inherits from
// Quad6DOFVarTimeRelative and overrides f() to sample the accel buffer at the
// 4 Runge–Kutta sub-steps. Jacobians use the midpoint accel as a frozen
// linearisation point ("frozen Jacobian" MPC pattern).
//
// propagate() (13-dim) is intentionally disabled; warm-start rollouts should
// use propagate14() so that IDX_DT tracks accumulated time within this solve.
template<typename Scalar>
class Quad6DOFVarTimeRelativeTV : public Quad6DOFVarTimeRelative<Scalar> {
 target_models::TargetAccelBuffer buf_;
 double t0_abs_;

public:
 explicit Quad6DOFVarTimeRelativeTV(const target_models::TargetAccelBuffer& buf,
																		double t0_abs)
	: Quad6DOFVarTimeRelative<Scalar>()
	, buf_(buf)
	, t0_abs_(t0_abs) {}

 void updateBuffer(const target_models::TargetAccelBuffer& buf) { buf_ = buf; }

 Vector<Scalar> f(const Vector<Scalar>& x,
									const Vector<Scalar>& u) const override {
	auto xd = x.segment(0, this->NX_PHYS).template cast<double>();
	auto ud = u.segment(0, this->NU_PHYS).template cast<double>();
	double Th   = static_cast<double>(u(IDX_THETA));
	double tabs = static_cast<double>(x(IDX_DT)) + t0_abs_;

	Eigen::Vector3d a1 = buf_.getAccel(tabs);
	auto k1 = this->xdot_impl(xd, ud, a1);

	Eigen::Vector3d a2 = buf_.getAccel(tabs + 0.5*Th);
	auto k2 = this->xdot_impl(xd + 0.5*Th*k1, ud, a2);

	Eigen::Vector3d a3 = buf_.getAccel(tabs + 0.5*Th);
	auto k3 = this->xdot_impl(xd + 0.5*Th*k2, ud, a3);

	Eigen::Vector3d a4 = buf_.getAccel(tabs + Th);
	auto k4 = this->xdot_impl(xd + Th*k3, ud, a4);

	Eigen::VectorXd xn = xd + (Th/6.0)*(k1 + 2*k2 + 2*k3 + k4);
	xn.segment(6,4).normalize();

	const_cast<Quad6DOFVarTimeRelativeTV*>(this)
	 ->setTargetAccel(buf_.getAccel(tabs + 0.5*Th));

	Vector<Scalar> res(NX_SS);
	res.segment(0, this->NX_PHYS) = xn.template cast<Scalar>();
	res(IDX_DT) = x(IDX_DT) + u(IDX_THETA);
	return res;
 }

 Eigen::VectorXd propagate(const Eigen::VectorXd&,
													 const Eigen::VectorXd&,
													 double) const {
	assert(false && "Quad6DOFVarTimeRelativeTV: use propagate14(); 13-dim propagate() is disabled");
	return Eigen::VectorXd();
 }

 Eigen::VectorXd propagate14(const Eigen::VectorXd& x14,
														 const Eigen::VectorXd& u_phys,
														 double Th) const {
	double tabs = x14(IDX_DT) + t0_abs_;
	auto xd = x14.segment(0, this->NX_PHYS);

	Eigen::Vector3d a1 = buf_.getAccel(tabs);
	auto k1 = this->xdot_impl(xd, u_phys, a1);
	Eigen::Vector3d a2 = buf_.getAccel(tabs + 0.5*Th);
	auto k2 = this->xdot_impl(xd + 0.5*Th*k1, u_phys, a2);
	Eigen::Vector3d a3 = buf_.getAccel(tabs + 0.5*Th);
	auto k3 = this->xdot_impl(xd + 0.5*Th*k2, u_phys, a3);
	Eigen::Vector3d a4 = buf_.getAccel(tabs + Th);
	auto k4 = this->xdot_impl(xd + Th*k3, u_phys, a4);

	Eigen::VectorXd xn14(NX_SS);
	xn14.segment(0, this->NX_PHYS) = xd + (Th/6.0)*(k1+2*k2+2*k3+k4);
	xn14.segment(6,4).normalize();
	xn14(IDX_DT) = x14(IDX_DT) + Th;
	return xn14;
 }
};

// ── Costs ─────────────────────────────────────────────────────────────────────
template<typename Scalar>
class TimeCost : public StageCostBase<Scalar> {
 Scalar eps_;
 Scalar w_vz_;
public:
 explicit TimeCost(double e=1e-4, double w_vz=1.0)
 : eps_(static_cast<Scalar>(e)), w_vz_(static_cast<Scalar>(w_vz)) {}
 Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
 return u(IDX_THETA) + eps_*(x.segment(3,3).squaredNorm()+x.segment(10,3).squaredNorm())
 + w_vz_*x(5)*x(5);
 }
 Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
 Vector<Scalar> g=Vector<Scalar>::Zero(NX_SS);
 g.segment(3,3) = Scalar(2)*eps_*x.segment(3,3);
 g.segment(10,3) = Scalar(2)*eps_*x.segment(10,3);
 g(5) += Scalar(2)*w_vz_*x(5);
 return g;
 }
 Vector<Scalar> qu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
 Vector<Scalar> g=Vector<Scalar>::Zero(NU_SS); g(IDX_THETA)=Scalar(1); return g;
 }
 Matrix<Scalar> qxx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
 Matrix<Scalar> H=Matrix<Scalar>::Zero(NX_SS,NX_SS);
 for(int i=3;i<6;++i) H(i,i)=Scalar(2)*eps_;
 for(int i=10;i<13;++i) H(i,i)=Scalar(2)*eps_;
 H(5,5) += Scalar(2)*w_vz_;
 return H;
 }
 Matrix<Scalar> quu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
 return Matrix<Scalar>::Zero(NU_SS,NU_SS); }
 Matrix<Scalar> qxu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
 return Matrix<Scalar>::Zero(NX_SS,NU_SS); }
};

template<typename Scalar>
class RelTermCost : public TerminalCostBase<Scalar> {
 double wp_;
 double wv_;
 double wvz_;
 double watt_;
 double wom_;
 double vz_ref_;
public:
 explicit RelTermCost(double wp=500.0, double wv=50.0,
 double wvz=100.0, double watt=200.0,
 double wom=50.0, double vz_ref=-0.1)
 : wp_(wp), wv_(wv), wvz_(wvz), watt_(watt), wom_(wom), vz_ref_(vz_ref) {}

 Scalar p(const Vector<Scalar>& x) const override {
 Scalar ep = x.template segment<2>(0).squaredNorm();
 Scalar ez = x(2)*x(2);
 Scalar evxy = x.template segment<2>(3).squaredNorm();
 Scalar evz = (x(5) - Scalar(vz_ref_)) * (x(5) - Scalar(vz_ref_));
 Scalar eatt = x(7)*x(7) + x(8)*x(8);
 Scalar eom = x.template segment<3>(10).squaredNorm();
 return Scalar(wp_)*(ep+ez)
 + Scalar(wv_)*evxy
 + Scalar(wvz_)*evz
 + Scalar(watt_)*eatt
 + Scalar(wom_)*eom;
 }

 Vector<Scalar> px(const Vector<Scalar>& x) const override {
 Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
 g.segment(0,2) = Scalar(2*wp_)*x.segment(0,2);
 g(2) = Scalar(2*wp_)*x(2);
 g.segment(3,2) = Scalar(2*wv_)*x.segment(3,2);
 g(5) = Scalar(2*wvz_)*(x(5) - Scalar(vz_ref_));
 g(7) = Scalar(2*watt_)*x(7);
 g(8) = Scalar(2*watt_)*x(8);
 g.segment(10,3) = Scalar(2*wom_)*x.segment(10,3);
 return g;
 }

 Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
 Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
 H(0,0)=H(1,1)=H(2,2) = Scalar(2*wp_);
 H(3,3)=H(4,4) = Scalar(2*wv_);
 H(5,5) = Scalar(2*wvz_);
 H(7,7)=H(8,8) = Scalar(2*watt_);
 H(10,10)=H(11,11)=H(12,12) = Scalar(2*wom_);
 return H;
 }
};

// ── Stage constraints ─────────────────────────────────────────────────────────
template<typename Scalar>
class FminCon:public StageConstraintBase<Scalar>{public:
 FminCon(){this->constraint_type=ConstraintType::NO;this->dim_c=1;}
 Vector<Scalar> c(const Vector<Scalar>&,const Vector<Scalar>& u)const override{
 return(Vector<Scalar>(1)<<Scalar(FMIN)-u(0)).finished();}
 Matrix<Scalar> cx(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
 return Matrix<Scalar>::Zero(1,x.size());}
 Matrix<Scalar> cu(const Vector<Scalar>&,const Vector<Scalar>& u)const override{
 Matrix<Scalar> J=Matrix<Scalar>::Zero(1,u.size());J(0,0)=Scalar(-1);return J;}};

template<typename Scalar>
class FmaxCon:public StageConstraintBase<Scalar>{public:
 FmaxCon(){this->constraint_type=ConstraintType::NO;this->dim_c=1;}
 Vector<Scalar> c(const Vector<Scalar>&,const Vector<Scalar>& u)const override{
 return(Vector<Scalar>(1)<<u(0)-Scalar(FMAX)).finished();}
 Matrix<Scalar> cx(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
 return Matrix<Scalar>::Zero(1,x.size());}
 Matrix<Scalar> cu(const Vector<Scalar>&,const Vector<Scalar>& u)const override{
 Matrix<Scalar> J=Matrix<Scalar>::Zero(1,u.size());J(0,0)=Scalar(1);return J;}};

template<typename Scalar>
class MomentCon:public StageConstraintBase<Scalar>{Scalar tau_;public:
 MomentCon():tau_(static_cast<Scalar>(TAU_MAX)){
 this->constraint_type=ConstraintType::SOC;this->dim_c=4;}
 Vector<Scalar> c(const Vector<Scalar>&,const Vector<Scalar>& u)const override{
 Vector<Scalar> cn(4);cn<<tau_,u(1),u(2),u(3);return -cn;}
 Matrix<Scalar> cx(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
 return Matrix<Scalar>::Zero(4,x.size());}
 Matrix<Scalar> cu(const Vector<Scalar>&,const Vector<Scalar>& u)const override{
 Matrix<Scalar> J=Matrix<Scalar>::Zero(4,u.size());
 J(1,1)=J(2,2)=J(3,3)=Scalar(-1);return J;}};

template<typename Scalar>
class ThetaBounds:public StageConstraintBase<Scalar>{Scalar lo_,hi_;public:
 ThetaBounds(double lo,double hi):lo_(static_cast<Scalar>(lo)),hi_(static_cast<Scalar>(hi)){
 this->constraint_type=ConstraintType::NO;this->dim_c=2;}
 Vector<Scalar> c(const Vector<Scalar>&,const Vector<Scalar>& u)const override{
 Vector<Scalar> cn(2);cn(0)=u(IDX_THETA)-hi_;cn(1)=lo_-u(IDX_THETA);return cn;}
 Matrix<Scalar> cx(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
 return Matrix<Scalar>::Zero(2,x.size());}
 Matrix<Scalar> cu(const Vector<Scalar>&,const Vector<Scalar>& u)const override{
 Matrix<Scalar> J=Matrix<Scalar>::Zero(2,u.size());
 J(0,IDX_THETA)=Scalar(1);J(1,IDX_THETA)=Scalar(-1);return J;}};

template<typename Scalar>
class ZFloorCon:public StageConstraintBase<Scalar>{public:
 ZFloorCon(){this->constraint_type=ConstraintType::NO;this->dim_c=1;}
 Vector<Scalar> c(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
 return(Vector<Scalar>(1)<<-x(2)).finished();}
 Matrix<Scalar> cx(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
 Matrix<Scalar> J=Matrix<Scalar>::Zero(1,x.size());J(0,2)=Scalar(-1);return J;}
 Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&)const override{
 return Matrix<Scalar>::Zero(1, 5);} };

template<typename Scalar>
class VzMinCon:public StageConstraintBase<Scalar>{Scalar vz_min_;public:
 explicit VzMinCon(double vz_min):vz_min_(static_cast<Scalar>(vz_min)){
 this->constraint_type=ConstraintType::NO;this->dim_c=1;}
 Vector<Scalar> c(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
 return(Vector<Scalar>(1)<<vz_min_-x(5)).finished();}
 Matrix<Scalar> cx(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
 Matrix<Scalar> J=Matrix<Scalar>::Zero(1,x.size());J(0,5)=Scalar(-1);return J;}
 Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&)const override{
 return Matrix<Scalar>::Zero(1, 5);} };

template<typename Scalar>
class GlideslopeCon_rel : public StageConstraintBase<Scalar> {
 Scalar tan_gs_;
public:
 explicit GlideslopeCon_rel(double tg=GS_TAN) : tan_gs_(static_cast<Scalar>(tg)) {
 this->constraint_type = ConstraintType::SOC;
 this->dim_c = 3;
 }
 Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
 Vector<Scalar> cn(3);
 cn(0) = tan_gs_ * x(2);
 cn(1) = x(0);
 cn(2) = x(1);
 return -cn;
 }
 Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
 Matrix<Scalar> J = Matrix<Scalar>::Zero(3, NX_SS);
 J(0,2) = -tan_gs_;
 J(1,0) = Scalar(-1.0);
 J(2,1) = Scalar(-1.0);
 return J;
 }
 Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
 return Matrix<Scalar>::Zero(3, NU_SS);
 }
};

// ── Factory ──────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
 const Eigen::VectorXd& x0_rel,
 const Eigen::VectorXd& /* terminal_state */,   // kept for interface compatibility
 const std::vector<Eigen::VectorXd>& prev_U = {},
 const std::vector<Eigen::VectorXd>& prev_X = {},
 const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero(),
 const std::vector<Eigen::MatrixXd>& prev_K = {},
 const target_models::TargetAccelBuffer& buf_in = {},
 double t0_abs = 0.0)
{
 auto problem = std::make_shared<OptimalControlProblem<double>>(HORIZON);

 // Use externally prepared buffer/time origin when provided. Fall back to a
 // constant target accel so existing callers keep previous behaviour.
 target_models::TargetAccelBuffer buf = buf_in;
 if (buf.accels.empty()) {
  buf.t_start = t0_abs;
  buf.dt = TH_INIT;
  buf.accels = {target_accel};
 }

 auto dyn = std::make_shared<Quad6DOFVarTimeRelativeTV<double>>(buf, t0_abs);
 dyn->setMass(MASS);
 dyn->setGravity(GRAVITY);
 dyn->setJb(J_B);
 dyn->setTargetAccel(buf.getAccel(t0_abs));

 auto cost = std::make_shared<TimeCost<double>>(1e-4, 1.0);
 auto tcost = std::make_shared<RelTermCost<double>>(500.0, 50.0);
 auto cfmin = std::make_shared<FminCon<double>>();
 auto cfmax = std::make_shared<FmaxCon<double>>();
 auto cmom = std::make_shared<MomentCon<double>>();
 auto cth = std::make_shared<ThetaBounds<double>>(THL, THH);
 auto czfl = std::make_shared<ZFloorCon<double>>();
 auto cvz = std::make_shared<VzMinCon<double>>(-VZ_LAND_MAX);
 auto cgs = std::make_shared<GlideslopeCon_rel<double>>();

 for(int k=0; k<HORIZON; ++k){
  problem->setStageDynamics(k, dyn);
  problem->setStageCost(k, cost);
  problem->addStageConstraint(k, cfmin);
  problem->addStageConstraint(k, cfmax);
  problem->addStageConstraint(k, cmom);
  problem->addStageConstraint(k, cth);
  problem->addStageConstraint(k, czfl);
  problem->addStageConstraint(k, cvz);
  problem->addStageConstraint(k, cgs);
 }
 problem->setTerminalCost(tcost);

 // Initial state: IDX_DT = 0 so time accumulator starts from zero each solve.
 Eigen::VectorXd x0ss(NX_SS);
 x0ss.segment(0,NX) = x0_rel;
 x0ss(IDX_DT)       = 0.0;
 problem->setInitialState(0, x0ss);

 // Warm-start rollout in 14‑dim space so IDX_DT is tracked consistently with
 // the time-varying dynamics implementation.
 Eigen::VectorXd sim14(NX_SS);
 sim14.segment(0,NX) = x0_rel;
 sim14(IDX_DT)       = 0.0;

 bool use_feedback = (!prev_U.empty() && !prev_X.empty() && !prev_K.empty()
					  && static_cast<int>(prev_U.size()) >= HORIZON
					  && static_cast<int>(prev_X.size()) >  HORIZON
					  && static_cast<int>(prev_K.size()) >= HORIZON);

 for(int k=0; k<HORIZON; ++k){
  Eigen::VectorXd u0(NU_SS); u0.setZero();

  if (use_feedback) {
   // Closed-loop DDP policy: u = prev_U[k] + K[k]*(sim_ss - prev_X[k])
   Eigen::VectorXd delta = sim14 - prev_X[k];
   u0 = prev_U[k] + prev_K[k] * delta;
   u0(0)         = std::max((double)FMIN, std::min((double)FMAX, (double)u0(0)));
   u0(1)         = std::max(-TAU_MAX, std::min(TAU_MAX, (double)u0(1)));
   u0(2)         = std::max(-TAU_MAX, std::min(TAU_MAX, (double)u0(2)));
   u0(3)         = std::max(-TAU_MAX, std::min(TAU_MAX, (double)u0(3)));
   u0(IDX_THETA) = std::max(THL,      std::min(THH,      (double)u0(IDX_THETA)));
  } else if (!prev_U.empty() && k < static_cast<int>(prev_U.size())) {
   u0 = prev_U[k];
  } else {
   // Geometric straight-line warm-start, matching the original stateswitch
   // behaviour but with an accel offset so that thrust accounts for
   // target_accel as well as gravity.
   Eigen::Vector3d p0 = sim14.head(3);
   Eigen::Vector3d v0 = sim14.segment(3,3);
   double T_g = (HORIZON - k) * TH_INIT;
   if (T_g < TH_INIT) T_g = TH_INIT;
   Eigen::Vector3d a_req = 2.0*(-p0 - v0*T_g) / (T_g*T_g);
   const double a_req_z_min = (-VZ_LAND_MAX - v0.z()) / T_g;
   a_req.z() = std::max(a_req.z(), a_req_z_min);

   // Subtract target_accel so the warm-start thrust includes centripetal
   // terms if the planner later supplies a non-zero vector.
   Eigen::Vector3d fw = MASS*(a_req - target_accel - GRAVITY);
   double fz_g = std::max((double)FMIN, std::min((double)FMAX, fw.norm()));
   u0(0) = fz_g;
   u0(IDX_THETA) = TH_INIT;
  }

  problem->setInitialControl(k, u0);

  // 14‑dim propagate with IDX_DT tracking; clamp altitude to keep the
  // warm-start above the floor.
  Eigen::VectorXd sim14_next = dyn->propagate14(sim14, u0.segment(0,NU), u0(IDX_THETA));
  sim14_next(2) = std::max(sim14_next(2), 0.0);
  problem->setInitialState(k+1, sim14_next);
  sim14 = sim14_next;
 }

 return problem;
}

inline OCPDescriptor descriptor() {
    OCPDescriptor d;
    d.name = "stateswitch";
    d.dt = TH_INIT;
    d.default_n_replay = DEFAULT_N_REPLAY;
    d.default_mass_kg = DEFAULT_MASS_KG;
    d.warm_start = OCPDescriptor::WarmStart::Feedback;
    d.command_mode = OCPDescriptor::CommandMode::CmdFullState;
    d.drone_odom_mode = OCPDescriptor::DroneOdomMode::Absolute;
    d.variable_dt = true;
    d.state_dim = 13;
    d.control_dim = 4;
    d.skip_altitude_validation = true;
    d.log_state_headers = OCPLoggerDefaults::getStateHeaders13D();
    d.state_names = OCPLoggerDefaults::getStateNames13D();
    d.control_names = OCPLoggerDefaults::getControlNames4D();
    d.extract_actual_state_row = OCPLoggerDefaults::getActualStateRow13D;
    d.make_hover_state = OCPLoggerDefaults::makeHoverState13D;
    d.transform_state = [](const Eigen::VectorXd& x, const TargetSnapshot& t) {
        if (t.valid) {
            Eigen::VectorXd xr = x;
            xr.segment(0, 3) -= t.position;
            xr.segment(3, 3) -= t.velocity;
            return xr;
        }
        return x;
    };
    d.post_process_result = [](SolverResult& r, const TargetSnapshot& t) {
        r.is_relative_plan = t.valid;
        if (t.valid) {
            r.target_snapshot_pos = t.position;
            r.target_snapshot_vel = t.velocity;
            r.target_snapshot_acc = t.acceleration;
        } else {
            r.target_snapshot_pos.setZero();
            r.target_snapshot_vel.setZero();
            r.target_snapshot_acc.setZero();
        }
    };
    d.prepare_extra = [](const PlannerConfig& cfg, double t_abs, const TargetSnapshot&) {
        StateswitchExtra ex;
        ex.t0_abs = t_abs;
        if (cfg.target_accel_buffer.has_value() && !cfg.target_accel_buffer->accels.empty()) {
            ex.buf = cfg.target_accel_buffer.value();
        } else {
            target_models::CircularTarget circ;
            circ.center << cfg.circle_center_x, cfg.circle_center_y, cfg.circle_center_z;
            circ.R = cfg.circle_R;
            circ.omega = cfg.circle_omega;
            circ.phi0 = cfg.circle_phi0 - cfg.circle_omega * cfg.t_start_abs;
            const double buf_dur = HORIZON * THH + 1.0;
            ex.buf.populateFromModel(circ, t_abs, buf_dur, 0.05);
        }
        return std::any(ex);
    };
    d.reconstruct_world_state = [](const Eigen::VectorXd& x, const TargetSnapshot& t) {
        Eigen::VectorXd xw = x;
        if (t.valid && xw.size() >= 6) {
            xw.segment(0, 3) += t.position;
            xw.segment(3, 3) += t.velocity;
        }
        return xw;
    };
    d.getSolverParams = getSolverParams;
    d.create = [](const OCPCreateArgs& a) {
        StateswitchExtra ex;
        if (a.extra.has_value()) {
            try { ex = std::any_cast<StateswitchExtra>(a.extra); }
            catch (const std::bad_any_cast&) {}
        }
        return create(a.current_state, a.terminal_state,
                      a.prev_U, a.prev_X, a.target_accel, a.prev_K,
                      ex.buf, ex.t0_abs);
    };
    return d;
}

} // namespace StateswitchOCP
