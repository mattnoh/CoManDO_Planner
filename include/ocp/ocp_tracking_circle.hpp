/// @file ocp_tracking_circle.hpp
/// @brief Single-shot OCP for quadrotor landing on a moving circular target.
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
///   vz is NOT constrained, allowing controlled descent via RelTermCost.
///
/// • Terminal cost: RelTermCost — soft penalties on rz, (vz-vz_ref), attitude,
///   and angular velocity. Prevents over-constraining the problem.
///
/// • Stage cost: TimeCost with vz_ref tracking throughout trajectory.
///
/// • Output: Absolute coordinates (converted from relative frame internally).
///
/// EXECUTION MODE: Open-loop only.
/// ─────────────────────────────────────────────────────────────────────────

#pragma once

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

namespace TrackingCircleOCP {

// ── Dimensions ────────────────────────────────────────────────────────────────
static constexpr int NX = 13;
static constexpr int NU = 4;
static constexpr int NX_SS = 14;
static constexpr int NU_SS = 5;
static constexpr int IDX_DT = 13;
static constexpr int IDX_THETA = 4;

// ── Horizon / timing ─────────────────────────────────────────────────────────
static constexpr int N = 80;
static constexpr int TH_INIT = 0.1;
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

// ── Circular target ───────────────────────────────────────────────────────────
struct CircularTarget {
    Eigen::Vector3d center;
    double R, omega, phi0;

    Eigen::Vector3d pos(double t) const {
        double ph = omega*t + phi0;
        return center + Eigen::Vector3d(R*std::cos(ph), R*std::sin(ph), 0.0);
    }
    Eigen::Vector3d vel(double t) const {
        double ph = omega*t + phi0;
        return Eigen::Vector3d(-R*omega*std::sin(ph), R*omega*std::cos(ph), 0.0);
    }
    Eigen::Vector3d accel(double t) const {
        double ph = omega*t + phi0;
        return Eigen::Vector3d(-R*omega*omega*std::cos(ph),
                               -R*omega*omega*std::sin(ph), 0.0);
    }
    Eigen::VectorXd state(double t) const {
        Eigen::VectorXd s(6); s.head(3)=pos(t); s.tail(3)=vel(t); return s;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Quad6DOFVarTimeRelativeTV — time-varying target acceleration
// ─────────────────────────────────────────────────────────────────────────────
template<typename Scalar>
class Quad6DOFVarTimeRelativeTV : public Quad6DOFVarTimeRelative<Scalar> {
    CircularTarget circ_;
    double t0_abs_;

public:
    Quad6DOFVarTimeRelativeTV(const CircularTarget& circ, double t0_abs)
        : Quad6DOFVarTimeRelative<Scalar>(), circ_(circ), t0_abs_(t0_abs) {}

    Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto xd = x.segment(0, this->NX_PHYS).template cast<double>();
        auto ud = u.segment(0, this->NU_PHYS).template cast<double>();
        double Th = static_cast<double>(u(IDX_THETA));
        double tabs = static_cast<double>(x(IDX_DT)) + t0_abs_;

        Eigen::Vector3d a1 = circ_.accel(tabs);
        auto k1 = this->xdot_impl(xd, ud, a1);

        Eigen::Vector3d a2 = circ_.accel(tabs + 0.5*Th);
        auto k2 = this->xdot_impl(xd + 0.5*Th*k1, ud, a2);

        Eigen::Vector3d a3 = circ_.accel(tabs + 0.5*Th);
        auto k3 = this->xdot_impl(xd + 0.5*Th*k2, ud, a3);

        Eigen::Vector3d a4 = circ_.accel(tabs + Th);
        auto k4 = this->xdot_impl(xd + Th*k3, ud, a4);

        Eigen::VectorXd xn = xd + (Th/6.0)*(k1 + 2*k2 + 2*k3 + k4);
        xn.segment(6, 4).normalize();

        const_cast<Quad6DOFVarTimeRelativeTV*>(this)
            ->setTargetAccel(circ_.accel(tabs + 0.5*Th));

        Vector<Scalar> res(14);
        res.segment(0, this->NX_PHYS) = xn.template cast<Scalar>();
        res(IDX_DT) = x(IDX_DT) + u(IDX_THETA);
        return res;
    }

    Eigen::VectorXd propagate14(const Eigen::VectorXd& x14,
                                 const Eigen::VectorXd& u_phys,
                                 double Th) const {
        double tabs = x14(IDX_DT) + t0_abs_;
        auto xd = x14.segment(0, this->NX_PHYS);

        Eigen::Vector3d a1 = circ_.accel(tabs);
        auto k1 = this->xdot_impl(xd, u_phys, a1);
        Eigen::Vector3d a2 = circ_.accel(tabs + 0.5*Th);
        auto k2 = this->xdot_impl(xd + 0.5*Th*k1, u_phys, a2);
        Eigen::Vector3d a3 = circ_.accel(tabs + 0.5*Th);
        auto k3 = this->xdot_impl(xd + 0.5*Th*k2, u_phys, a3);
        Eigen::Vector3d a4 = circ_.accel(tabs + Th);
        auto k4 = this->xdot_impl(xd + Th*k3, u_phys, a4);

        Eigen::VectorXd xn14(14);
        xn14.segment(0, this->NX_PHYS) = xd + (Th/6.0)*(k1+2*k2+2*k3+k4);
        xn14.segment(6, 4).normalize();
        xn14(IDX_DT) = x14(IDX_DT) + Th;
        return xn14;
    }
};

// ── TimeCost ──────────────────────────────────────────────────────────────────
template<typename Scalar>
class TimeCost : public StageCostBase<Scalar> {
    Scalar eps_, w_vz_, vz_ref_;
public:
    explicit TimeCost(double e=1e-4, double w_vz=0.5, double vz_ref=VZ_REF)
        : eps_(static_cast<Scalar>(e)),
          w_vz_(static_cast<Scalar>(w_vz)),
          vz_ref_(static_cast<Scalar>(vz_ref)) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Scalar dvz = x(5) - vz_ref_;
        return u(IDX_THETA)
             + eps_*(x.segment(3,3).squaredNorm()+x.segment(10,3).squaredNorm())
             + w_vz_*dvz*dvz;
    }
    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Vector<Scalar> g=Vector<Scalar>::Zero(NX_SS);
        g.segment(3,3) = Scalar(2)*eps_*x.segment(3,3);
        g.segment(10,3) = Scalar(2)*eps_*x.segment(10,3);
        g(5) += Scalar(2)*w_vz_*(x(5) - vz_ref_);
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
        return Matrix<Scalar>::Zero(NU_SS,NU_SS);
    }
    Matrix<Scalar> qxu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(NX_SS,NU_SS);
    }
};

// ── RelTermCost ───────────────────────────────────────────────────────────────
template<typename Scalar>
class RelTermCost : public TerminalCostBase<Scalar> {
    double wz_, wvz_, watt_, wom_, vz_ref_;
public:
    explicit RelTermCost(double wz=1000.0, double wvz=2000.0,
                         double watt=200.0, double wom=100.0, double vz_ref=VZ_REF)
        : wz_(wz), wvz_(wvz), watt_(watt), wom_(wom), vz_ref_(vz_ref) {}

    Scalar p(const Vector<Scalar>& x) const override {
        Scalar ez = x(2)*x(2);
        Scalar evz = (x(5) - Scalar(vz_ref_)) * (x(5) - Scalar(vz_ref_));
        Scalar eatt = x(7)*x(7) + x(8)*x(8);
        Scalar eom = x.template segment<3>(10).squaredNorm();
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
    Matrix<Scalar> pxx(const Vector<Scalar>&) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(NX_SS, NX_SS);
        H(2,2) = Scalar(2*wz_);
        H(5,5) = Scalar(2*wvz_);
        H(7,7) = Scalar(2*watt_);
        H(8,8) = Scalar(2*watt_);
        H(10,10) = H(11,11) = H(12,12) = Scalar(2*wom_);
        return H;
    }
};

// ── RelInterceptCon ───────────────────────────────────────────────────────────
template<typename Scalar>
class RelInterceptCon : public TerminalConstraintBase<Scalar> {
public:
    RelInterceptCon() {
        this->constraint_type = ConstraintType::EQ;
        this->dim_cT = 5;
    }
    Vector<Scalar> cT(const Vector<Scalar>& x) const override {
        Vector<Scalar> c(5);
        c(0) = x(0); c(1) = x(1); c(2) = x(2);
        c(3) = x(3); c(4) = x(4);
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
inline Param getSolverParams() {
    Param p;
    p.reg1_min = 1e-2; p.reg2_min = 0.5; p.mu_mul = 0.1;
    p.rho = 5.0; p.rhoT = 50.0; p.rho_mul = 5.0;
    p.tolerance = 1e-3; p.max_iter = 1000;
    p.is_quaternion_in_state = false;
    return p;
}

// ── Factory function ──────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& x0_abs,
    const CircularTarget& circ,
    double t0_abs,
    double th_init = TH_INIT,
    double th_min = THL,
    double th_max = THH)
{
    auto problem = std::make_shared<OptimalControlProblem<double>>(N);

    auto dyn = std::make_shared<Quad6DOFVarTimeRelativeTV<double>>(circ, t0_abs);
    dyn->setMass(MASS);
    dyn->setGravity(GRAVITY);
    dyn->setJb(J_B);
    dyn->setTargetAccel(circ.accel(t0_abs));

    auto cost = std::make_shared<TimeCost<double>>(1e-4, 0.5, VZ_REF);
    auto tcost = std::make_shared<RelTermCost<double>>(1000.0, 2000.0, 200.0, 100.0, VZ_REF);
    auto cfmin = std::make_shared<FminCon<double>>();
    auto cfmax = std::make_shared<FmaxCon<double>>();
    auto cmom = std::make_shared<MomentCon<double>>();
    auto cth = std::make_shared<ThetaBounds<double>>(th_min, th_max);
    auto czfl = std::make_shared<ZFloorCon<double>>();
    auto cvz = std::make_shared<VzMinCon<double>>(-VZ_LAND_MAX);
    auto cgs = std::make_shared<GlideslopeCon_rel<double>>();
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

    Eigen::VectorXd x0_rel(NX);
    x0_rel.head(3) = x0_abs.head(3) - circ.pos(t0_abs);
    x0_rel.segment(3,3) = x0_abs.segment(3,3) - circ.vel(t0_abs);
    x0_rel.segment(6,7) = x0_abs.segment(6,7);

    Eigen::VectorXd x0ss(NX_SS);
    x0ss.segment(0,NX) = x0_rel;
    x0ss(IDX_DT) = 0.0;
    problem->setInitialState(0, x0ss);

    Eigen::VectorXd sim14(NX_SS);
    sim14.segment(0,NX) = x0_rel;
    sim14(IDX_DT) = 0.0;

    for(int k=0; k<N; ++k) {
        Eigen::VectorXd u0(NU_SS); u0.setZero();
        Eigen::Vector3d p0 = sim14.head(3);
        Eigen::Vector3d v0 = sim14.segment(3,3);
        double T_rem = (N - k) * th_init;
        if(T_rem < th_init) T_rem = th_init;
        Eigen::Vector3d a_req = 2.0*(-p0 - v0*T_rem) / (T_rem*T_rem);
        a_req.z() = std::max(a_req.z(), (-VZ_LAND_MAX - v0.z()) / T_rem);
        Eigen::Vector3d fw = MASS*(a_req - GRAVITY);
        double fz = std::max((double)FMIN, std::min((double)FMAX, fw.norm()));
        u0(0) = fz;
        u0(IDX_THETA) = th_init;

        problem->setInitialControl(k, u0);

        Eigen::VectorXd sim14_next = dyn->propagate14(sim14, u0.head(NU), th_init);
        sim14_next(2) = std::max(sim14_next(2), 0.0);
        problem->setInitialState(k+1, sim14_next);
        sim14 = sim14_next;
    }

    return problem;
}

// ── Convert relative trajectory to absolute coordinates ────────────────────────
inline std::vector<Eigen::VectorXd> convertToAbsolute(
    const std::vector<Eigen::VectorXd>& X_rel,
    const CircularTarget& tgt,
    double t0_abs)
{
    std::vector<Eigen::VectorXd> X_abs(X_rel.size());
    for(size_t k=0; k<X_rel.size(); ++k) {
        double DT = X_rel[k](IDX_DT);
        double tabs = t0_abs + DT;

        Eigen::VectorXd x_abs(NX);
        x_abs.head(3) = X_rel[k].head(3) + tgt.pos(tabs);
        x_abs.segment(3,3) = X_rel[k].segment(3,3) + tgt.vel(tabs);
        x_abs.segment(6,7) = X_rel[k].segment(6,7);
        X_abs[k] = x_abs;
    }
    return X_abs;
}

} // namespace TrackingCircleOCP
