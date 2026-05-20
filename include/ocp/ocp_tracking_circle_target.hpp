/// @file ocp_tracking_circle_target.hpp
/// @brief Single-shot OCP for quadrotor landing on a moving target.
///
/// Port of quad_cf_tracking_ol_circle_rel.cpp from ALIPDDP-main.
///
/// KEY FEATURES:
/// ─────────────────────────────────────────────────────────────────────────
/// • Single-shot solve (N=80). No receding horizon.
///
/// • Time-varying dynamics: Quad6DOFVarTimeRelativeTV reads x[IDX_DT] at each
///   RK4 k-point and computes circ.accel(t) analytically — the actual rotating
///   vector. Solves the "frozen acceleration" problem.
///
/// • Terminal constraint: RelInterceptCon (5-dim) — p_rel=0 (3) + vxy_rel=0 (2).
///   In relative-frame dynamics the target is always at the origin, so vxy_rel=0
///   simply means "arrive co-moving with the target in XY" — it does NOT force
///   the drone to orbit. vz is left to RelTermCost for controlled descent.
///
/// • Terminal cost: RelTermCost — soft penalties on rz, (vz-vz_ref), attitude,
///   and angular velocity.
///
/// • Stage cost: TimeCost — time-optimal + vz_ref tracking (w_vz=0.5) matching
///   the original cpp. No w_xy, no w_z (both fight the glideslope).
///
/// • Warm-start: pure geometric straight-line — states set directly to
///   p(k) = p0_rel*(N-k)/N, v(k) = v_desired (constant), quaternion = identity.
///   No dynamic propagation. This guarantees X[N] = (0,0,0,...) exactly so the
///   terminal constraint starts at zero violation. Works for any feasible p0_rel.
///
/// • Solver params: rho=15, rhoT=500, rho_mul=8.
///   rhoT raised from 100 — with N=80 stages the accumulated stage cost is ~80x
///   a single stage, so rhoT=100 is too weak to dominate the terminal constraint.
///
/// • Output: Relative coordinates. Planner-side integration reconstructs
///   absolute world-frame trajectories for publishing/logging.
///
/// EXECUTION MODE: Open-loop only.
/// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include "target/target_accel_buffer.hpp"
#include "planner_core/types.hpp"

#include "optimal_control_problem.h"
#include "dynamics/discrete_dynamics_base.h"
#include "dynamics/quad_6dof_dynamics_aug.h"
#include "cost/stage_cost_base.h"
#include "cost/terminal_cost_base.h"
#include "constraint/stage_constraint_base.h"
#include "constraint/terminal_constraint_base.h"
#include "alipddp/alipddp.h"

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <memory>

namespace TrackingCircleTargetOCP {

// ── Dimensions ────────────────────────────────────────────────────────────────
static constexpr int NX = 13;
static constexpr int NU = 4;
static constexpr int NX_SS = 14;
static constexpr int NU_SS = 5;
static constexpr int IDX_DT = 13;
static constexpr int IDX_THETA = 4;

// ── Horizon / timing ─────────────────────────────────────────────────────────
static constexpr int N = 80;
static constexpr double TH_INIT = 0.1;
static constexpr double THL = 0.05;
static constexpr double THH = 0.2;
static constexpr int DEFAULT_N_REPLAY = N;

// ── Vehicle parameters ────────────────────────────────────────────────────────
static constexpr double MASS = 0.027;
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
static const Eigen::Matrix3d J_B_INV = J_B.inverse();
static const Eigen::Vector3d GRAVITY(0.0, 0.0, -9.81);

// ── Glideslope ────────────────────────────────────────────────────────────────
static constexpr double GS_DEG = 60.0;
static const double GS_TAN = std::tan(GS_DEG * M_PI / 180.0);
static constexpr double VZ_LAND_MAX = 2.5;
static constexpr double VZ_REF = -0.15;

// ─────────────────────────────────────────────────────────────────────────────
// Quad6DOFVarTimeRelativeTV — time-varying target acceleration
// ─────────────────────────────────────────────────────────────────────────────
template<typename Scalar>
class Quad6DOFVarTimeRelativeTV : public Quad6DOFVarTimeRelative<Scalar> {
    target_models::TargetAccelBuffer buf_;
    double t0_abs_;

public:
    Quad6DOFVarTimeRelativeTV(const target_models::TargetAccelBuffer& buf, double t0_abs)
        : Quad6DOFVarTimeRelative<Scalar>(), buf_(buf), t0_abs_(t0_abs) {}

    Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto xd = x.segment(0, this->NX_PHYS).template cast<double>();
        auto ud = u.segment(0, this->NU_PHYS).template cast<double>();
        double Th = static_cast<double>(u(IDX_THETA));
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
        xn.segment(6, 4).normalize();

        const_cast<Quad6DOFVarTimeRelativeTV*>(this)
            ->setTargetAccel(buf_.getAccel(tabs + 0.5*Th));

        Vector<Scalar> res(14);
        res.segment(0, this->NX_PHYS) = xn.template cast<Scalar>();
        res(IDX_DT) = x(IDX_DT) + u(IDX_THETA);
        return res;
    }
};

// ── TimeCost ──────────────────────────────────────────────────────────────────
// Matches the original cpp exactly:
//   • u(IDX_THETA)            — minimize total time
//   • eps * (||v||² + ||ω||²) — numerical damping (1e-4)
//   • w_vz * (vz - vz_ref)²  — shapes descent rate throughout (w_vz=0.5)
//
// No w_xy (fights the glideslope as z→0 the allowed XY shrinks).
// No w_z  (directly antagonistic to GlideslopeCon_rel).
// No w_u  (cpp has zero quu; DDP converges fine with the vz term filling qxx(5,5)).
template<typename Scalar>
class TimeCost : public StageCostBase<Scalar> {
    Scalar eps_, w_vz_, vz_ref_;
public:
    explicit TimeCost(double e      = 1e-4,
                      double w_vz   = 0.5,
                      double vz_ref = VZ_REF)
        : eps_   (static_cast<Scalar>(e)),
          w_vz_  (static_cast<Scalar>(w_vz)),
          vz_ref_(static_cast<Scalar>(vz_ref)) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Scalar dvz = x(5) - vz_ref_;
        return u(IDX_THETA)
             + eps_*(x.segment(3,3).squaredNorm() + x.segment(10,3).squaredNorm())
             + w_vz_*dvz*dvz;
    }
    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(NX_SS);
        g.segment(3,3)  = Scalar(2)*eps_*x.segment(3,3);
        g(5)           += Scalar(2)*w_vz_*(x(5) - vz_ref_);
        g.segment(10,3) = Scalar(2)*eps_*x.segment(10,3);
        return g;
    }
    Vector<Scalar> qu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(NU_SS);
        g(IDX_THETA) = Scalar(1);
        return g;
    }
    Matrix<Scalar> qxx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(NX_SS, NX_SS);
        for(int i=3;  i<6;  ++i) H(i,i) = Scalar(2)*eps_;
        for(int i=10; i<13; ++i) H(i,i) = Scalar(2)*eps_;
        H(5,5) += Scalar(2)*w_vz_;
        return H;
    }
    Matrix<Scalar> quu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(NU_SS, NU_SS);
    }
    Matrix<Scalar> qxu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(NX_SS, NU_SS);
    }
};

// ── RelTermCost ───────────────────────────────────────────────────────────────
// Soft terminal costs on quantities NOT covered by the hard equality
// RelInterceptCon (which handles p_rel=0 and vxy_rel=0).
//   • wz   — soft pull on rz² (redundant with hard constraint, but cheap insurance)
//   • wvz  — steer vz toward vz_ref at touchdown
//   • watt — upright at touchdown (qx,qy → 0)
//   • wom  — not spinning at touchdown
template<typename Scalar>
class RelTermCost : public TerminalCostBase<Scalar> {
    double wz_, wvz_, watt_, wom_, vz_ref_;
public:
    explicit RelTermCost(double wz    = 1000.0,
                         double wvz   = 2000.0,
                         double watt  = 200.0,
                         double wom   = 100.0,
                         double vz_ref = VZ_REF)
        : wz_(wz), wvz_(wvz), watt_(watt), wom_(wom), vz_ref_(vz_ref) {}

    Scalar p(const Vector<Scalar>& x) const override {
        Scalar ez   = x(2)*x(2);
        Scalar evz  = (x(5) - Scalar(vz_ref_)) * (x(5) - Scalar(vz_ref_));
        Scalar eatt = x(7)*x(7) + x(8)*x(8);
        Scalar eom  = x.template segment<3>(10).squaredNorm();
        return Scalar(wz_)*ez + Scalar(wvz_)*evz + Scalar(watt_)*eatt + Scalar(wom_)*eom;
    }
    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g(2) = Scalar(2*wz_) * x(2);
        g(5) = Scalar(2*wvz_) * (x(5) - Scalar(vz_ref_));
        g(7) = Scalar(2*watt_) * x(7);
        g(8) = Scalar(2*watt_) * x(8);
        g.segment(10,3) = Scalar(2*wom_) * x.segment(10,3);
        return g;
    }
    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H(2,2)   = Scalar(2*wz_);
        H(5,5)   = Scalar(2*wvz_);
        H(7,7)   = Scalar(2*watt_);
        H(8,8)   = Scalar(2*watt_);
        H(10,10) = H(11,11) = H(12,12) = Scalar(2*wom_);
        return H;
    }
};

// ── RelInterceptCon ───────────────────────────────────────────────────────────
// 5-dim hard equality (matches original cpp): p_rel=0 (3) + vxy_rel=0 (2).
//
// vxy_rel=0 in the RELATIVE frame means the drone arrives co-moving with the
// target in XY — exactly what you want for a landing. It does NOT force the
// drone to orbit. vz is left to RelTermCost (soft) so the solver can arrive
// with a controlled non-zero sink rate rather than a forced dead stop.
template<typename Scalar>
class RelInterceptCon : public TerminalConstraintBase<Scalar> {
public:
    RelInterceptCon() {
        this->constraint_type = ConstraintType::EQ;
        this->dim_cT = 5;
    }
    Vector<Scalar> cT(const Vector<Scalar>& x) const override {
        Vector<Scalar> c(5);
        c(0) = x(0); c(1) = x(1); c(2) = x(2);  // p_rel → 0
        c(3) = x(3); c(4) = x(4);                 // vxy_rel → 0
        return c;
    }
    Matrix<Scalar> cTx(const Vector<Scalar>&) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(5, NX_SS);
        J(0,0)=Scalar(1); J(1,1)=Scalar(1); J(2,2)=Scalar(1);
        J(3,3)=Scalar(1); J(4,4)=Scalar(1);
        return J;
    }
};

// ── Stage constraints ─────────────────────────────────────────────────────────
template<typename Scalar>
class FminCon : public StageConstraintBase<Scalar> {
public:
    FminCon() { this->constraint_type=ConstraintType::NO; this->dim_c=1; }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return (Vector<Scalar>(1) << Scalar(FMIN)-u(0)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(1,NX_SS);
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(1,NU_SS); J(0,0)=Scalar(-1); return J;
    }
};

template<typename Scalar>
class FmaxCon : public StageConstraintBase<Scalar> {
public:
    FmaxCon() { this->constraint_type=ConstraintType::NO; this->dim_c=1; }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return (Vector<Scalar>(1) << u(0)-Scalar(FMAX)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(1,NX_SS);
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(1,NU_SS); J(0,0)=Scalar(1); return J;
    }
};

template<typename Scalar>
class MomentCon : public StageConstraintBase<Scalar> {
    Scalar tau_;
public:
    MomentCon() : tau_(static_cast<Scalar>(TAU_MAX)) {
        this->constraint_type=ConstraintType::SOC; this->dim_c=4;
    }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Vector<Scalar> cn(4); cn << tau_, u(1), u(2), u(3); return -cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(4,NX_SS);
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(4,NU_SS);
        J(1,1)=J(2,2)=J(3,3)=Scalar(-1); return J;
    }
};

template<typename Scalar>
class ThetaBounds : public StageConstraintBase<Scalar> {
    Scalar lo_, hi_;
public:
    ThetaBounds(double lo, double hi) : lo_(lo), hi_(hi) {
        this->constraint_type=ConstraintType::NO; this->dim_c=2;
    }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Vector<Scalar> cn(2); cn(0)=u(IDX_THETA)-hi_; cn(1)=lo_-u(IDX_THETA); return cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(2,NX_SS);
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(2,NU_SS);
        J(0,IDX_THETA)=Scalar(1); J(1,IDX_THETA)=Scalar(-1); return J;
    }
};

template<typename Scalar>
class ZFloorCon : public StageConstraintBase<Scalar> {
public:
    ZFloorCon() { this->constraint_type=ConstraintType::NO; this->dim_c=1; }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return (Vector<Scalar>(1) << -x(2)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(1,NX_SS); J(0,2)=Scalar(-1); return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(1,NU_SS);
    }
};

template<typename Scalar>
class VzMinCon : public StageConstraintBase<Scalar> {
    Scalar vz_min_;
public:
    explicit VzMinCon(double vz_min) : vz_min_(vz_min) {
        this->constraint_type=ConstraintType::NO; this->dim_c=1;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return (Vector<Scalar>(1) << vz_min_-x(5)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(1,NX_SS); J(0,5)=Scalar(-1); return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(1,NU_SS);
    }
};

template<typename Scalar>
class GlideslopeCon_rel : public StageConstraintBase<Scalar> {
    Scalar tan_gs_;
public:
    explicit GlideslopeCon_rel(double tg=GS_TAN) : tan_gs_(tg) {
        this->constraint_type=ConstraintType::SOC; this->dim_c=3;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Vector<Scalar> cn(3);
        cn(0) = tan_gs_*x(2); cn(1) = x(0); cn(2) = x(1);
        return -cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, NX_SS);
        J(0,2)=-tan_gs_; J(1,0)=Scalar(-1); J(2,1)=Scalar(-1);
        return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(3, NU_SS);
    }
};

// ── Solver params ─────────────────────────────────────────────────────────────
// rhoT raised to 500: with N=80 stages the accumulated stage cost is ~80x a
// single stage. rhoT=100 was too weak to dominate the terminal constraint
// against that accumulated cost — the solver reported KKT convergence while
// the terminal violation was still meters off.
inline Param getSolverParams() {
    Param p;
    p.reg1_min = 1e-2; p.reg2_min = 0.5; p.mu_mul = 0.1;
    p.rho      = 15.0;
    p.rhoT     = 500.0;  // was 100 — raised to enforce terminal constraint
    p.rho_mul  = 8.0;
    p.tolerance = 2e-3;
    p.max_iter  = 1000;
    p.is_quaternion_in_state = false;
    return p;
}

// ── Factory function ──────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& x0_rel,
    const target_models::TargetAccelBuffer& buf,
    double t0_abs,
    double th_init = TH_INIT,
    double th_min = THL,
    double th_max = THH)
{
    auto problem = std::make_shared<OptimalControlProblem<double>>(N);

    auto dyn = std::make_shared<Quad6DOFVarTimeRelativeTV<double>>(buf, t0_abs);
    dyn->setMass(MASS);
    dyn->setGravity(GRAVITY);
    dyn->setJb(J_B);
    dyn->setTargetAccel(buf.getAccel(t0_abs));

    auto cost  = std::make_shared<TimeCost<double>>(1e-4, 0.5, VZ_REF);
    auto tcost = std::make_shared<RelTermCost<double>>(1000.0, 2000.0, 200.0, 100.0, VZ_REF);

    auto cfmin = std::make_shared<FminCon<double>>();
    auto cfmax = std::make_shared<FmaxCon<double>>();
    auto cmom  = std::make_shared<MomentCon<double>>();
    auto cth   = std::make_shared<ThetaBounds<double>>(th_min, th_max);
    auto czfl  = std::make_shared<ZFloorCon<double>>();
    auto cvz   = std::make_shared<VzMinCon<double>>(-VZ_LAND_MAX);
    auto cgs   = std::make_shared<GlideslopeCon_rel<double>>();
    auto cterm = std::make_shared<RelInterceptCon<double>>();

    for(int k=0; k<N; ++k) {
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
    problem->addTerminalConstraint(cterm);

    Eigen::VectorXd x0ss(NX_SS);
    x0ss.segment(0,NX) = x0_rel;
    x0ss(IDX_DT) = 0.0;
    problem->setInitialState(0, x0ss);

    // ── Warm-start: pure geometric straight-line ──────────────────────────────
    //
    // This is a single-shot solve — there is no receding horizon, so the
    // warm-start only exists to put DDP in the basin of a good local minimum.
    //
    // States are set directly to the geometric straight line from p0_rel to the
    // origin — no dynamic propagation. This guarantees:
    //   • Every state is exactly on the straight line (cone-feasible by convexity)
    //   • X[N].head(5) = 0 exactly — terminal constraint starts at zero violation
    //   • Works for any p0_rel inside the cone, no dependence on a specific start
    //
    // Controls are set to hover thrust — just enough to keep the solver from
    // starting with wildly infeasible inputs. DDP will move them immediately.

    const double T_total = N * th_init;
    const Eigen::Vector3d p0_rel_ws = x0_rel.head(3);
    const Eigen::Vector3d v_desired = -p0_rel_ws / T_total;  // constant velocity to arrive at origin

    for(int k=0; k<N; ++k) {
        // State: interpolate straight line
        double frac = static_cast<double>(N - k) / N;
        Eigen::VectorXd xk(NX_SS);
        xk.setZero();
        xk.head(3)      = p0_rel_ws * frac;
        xk.segment(3,3) = v_desired;
        xk(6)           = 1.0;           // quaternion w=1, identity rotation
        xk(IDX_DT)      = k * th_init;
        problem->setInitialState(k, xk);

        // Control: hover thrust, nominal timestep
        Eigen::VectorXd u0(NU_SS);
        u0.setZero();
        u0(0)         = MASS * 9.81;
        u0(IDX_THETA) = th_init;
        problem->setInitialControl(k, u0);
    }

    // Terminal state: exactly at origin
    Eigen::VectorXd xN(NX_SS);
    xN.setZero();
    xN(6)      = 1.0;
    xN(IDX_DT) = N * th_init;
    problem->setInitialState(N, xN);

    return problem;
}

struct TrackingCircleTargetExtra {
    target_models::TargetAccelBuffer buf;
    double t_abs = 0.0;
};

inline OCPDescriptor descriptor() {
    OCPDescriptor d;
    d.name = "tracking_circle_target";
    d.dt = TH_INIT;
    d.default_n_replay = DEFAULT_N_REPLAY;
    d.default_mass_kg = MASS;
    d.warm_start = OCPDescriptor::WarmStart::Shift;
    d.command_mode = OCPDescriptor::CommandMode::CmdFullState;
    d.drone_odom_mode = OCPDescriptor::DroneOdomMode::AbsoluteShiftedTarget;
    d.needs_target_trajectory = true;
    d.disarm_on_landing_finish = true;
    d.state_dim = 13;
    d.control_dim = 4;
    d.skip_altitude_validation = true;
    d.log_state_headers = OCPLoggerDefaults::getStateHeaders13D();
    d.state_names = OCPLoggerDefaults::getStateNames13D();
    d.control_names = OCPLoggerDefaults::getControlNames4D();
    d.extract_actual_state_row = OCPLoggerDefaults::getActualStateRow13D;
    d.make_hover_state = OCPLoggerDefaults::makeHoverState13D;
    d.transform_state = [](const Eigen::VectorXd& x, const TargetSnapshot& t) {
        Eigen::VectorXd xr = x;
        xr.segment(0, 3) -= t.position;
        xr.segment(3, 3) -= t.velocity;
        return xr;
    };
    d.validate_target = [](const TargetSnapshot& t, double now_sec, double max_age) {
        if (t.odom_stamp_sec <= 0.0) return false;
        const double age = now_sec - t.odom_stamp_sec;
        const double kClockTol = 0.001;
        return !(age < -kClockTol || age >= max_age);
    };
    d.post_process_result = [](SolverResult& r, const TargetSnapshot& t) {
        r.is_relative_plan = true;
        if (t.odom_stamp_sec > 0.0) {
            r.target_snapshot_pos = t.position;
            r.target_snapshot_vel = t.velocity;
        } else {
            r.target_snapshot_pos.setZero();
            r.target_snapshot_vel.setZero();
        }
        r.target_snapshot_acc = t.acceleration;
    };
    d.prepare_extra = [](const PlannerConfig& cfg, double t_abs, const TargetSnapshot&) {
        TrackingCircleTargetExtra ex;
        ex.t_abs = t_abs;
        if (cfg.target_accel_buffer.has_value()) {
            ex.buf = cfg.target_accel_buffer.value();
        }
        return std::any(ex);
    };
    d.getSolverParams = getSolverParams;
    d.create = [](const OCPCreateArgs& a) {
        if (a.extra.has_value()) {
            try {
                auto ex = std::any_cast<TrackingCircleTargetExtra>(a.extra);
                return create(a.current_state, ex.buf, ex.t_abs);
            } catch (const std::bad_any_cast&) {}
        }
        return create(a.current_state, target_models::TargetAccelBuffer{}, a.t_abs);
    };
    return d;
}

} // namespace TrackingCircleTargetOCP
