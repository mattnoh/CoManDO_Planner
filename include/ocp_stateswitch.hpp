/// @file ocp_stateswitch.hpp
/// @brief OCP formulation for relative-frame landing on a moving target (state-switch).
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include "optimal_control_problem.h"

namespace StateswitchOCP {

// ── Dimensions ────────────────────────────────────────────────────────────────
static constexpr int NX     = 13;
static constexpr int NU     = 4;
static constexpr int NX_SS  = 14;   // NX + IDX_DT
static constexpr int NU_SS  = 5;    // NU + IDX_THETA
static constexpr int IDX_DT    = 13;
static constexpr int IDX_THETA = 4;

// ── Horizon / timing ─────────────────────────────────────────────────────────
static constexpr int    HORIZON = 30;
static constexpr double TH_INIT = 0.1;
static constexpr double THL     = 0.05;
static constexpr double THH     = 0.2;

// ── Vehicle parameters ───────────────────────────────────────────────────────
static constexpr double MASS    = 0.027;
static constexpr double IXX     = 1.66e-5;
static constexpr double IYY     = 1.66e-5;
static constexpr double IZZ     = 2.92e-5;
static constexpr double J_SCALE = 1.0 / IXX;
static constexpr double FMIN    = 0.08;
static constexpr double FMAX    = 0.6;
static constexpr double L_ARM   = 0.046;
static const     double TAU_MAX = L_ARM * (FMAX/4.0 - FMIN/4.0) * J_SCALE;

static const Eigen::Matrix3d J_B = [](){
    Eigen::Matrix3d J; J.setZero();
    J(0,0)=IXX*J_SCALE; J(1,1)=IYY*J_SCALE; J(2,2)=IZZ*J_SCALE;
    return J;
}();
static const Eigen::Vector3d GRAVITY(0.0, 0.0, -9.81);

// ── Glideslope parameters ─────────────────────────────────────────────────────
static constexpr double GS_DEG      = 60.0;
static const     double GS_TAN      = std::tan(GS_DEG * M_PI / 180.0);
static constexpr double VZ_LAND_MAX = 2.5; // m/s descent cap

// ── Costs ─────────────────────────────────────────────────────────────────────
// Time-optimal stage cost (same as rh_landing).
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
        g.segment(3,3)  = Scalar(2)*eps_*x.segment(3,3);
        g.segment(10,3) = Scalar(2)*eps_*x.segment(10,3);
        g(5) += Scalar(2)*w_vz_*x(5);
        return g;
    }
    Vector<Scalar> qu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Vector<Scalar> g=Vector<Scalar>::Zero(NU_SS); g(IDX_THETA)=Scalar(1); return g;
    }
    Matrix<Scalar> qxx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> H=Matrix<Scalar>::Zero(NX_SS,NX_SS);
        for(int i=3;i<6;++i)  H(i,i)=Scalar(2)*eps_;
        for(int i=10;i<13;++i) H(i,i)=Scalar(2)*eps_;
        H(5,5) += Scalar(2)*w_vz_;
        return H;
    }
    Matrix<Scalar> quu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(NU_SS,NU_SS); }
    Matrix<Scalar> qxu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(NX_SS,NU_SS); }
};

// Terminal cost in RELATIVE frame.
template<typename Scalar>
class RelTermCost : public TerminalCostBase<Scalar> {
    double wp_;     // position weight
    double wv_;     // xy-velocity weight
    double wvz_;    // vz weight (drives to vz_ref, not zero)
    double watt_;   // attitude tilt weight (qx,qy → 0)
    double wom_;    // body rate weight (omega → 0)
    double vz_ref_; // target descent rate in relative frame (m/s, negative = descending)
public:
    explicit RelTermCost(double wp=500.0, double wv=50.0,
                         double wvz=100.0, double watt=200.0,
                         double wom=50.0,  double vz_ref=-0.1)
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
        g(2)           = Scalar(2*wp_)*x(2);
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
    ThetaBounds(double lo,double hi):lo_(lo),hi_(hi){
        this->constraint_type=ConstraintType::NO;this->dim_c=2;}
    Vector<Scalar> c(const Vector<Scalar>&,const Vector<Scalar>& u)const override{
        Vector<Scalar> cn(2);cn(0)=u(IDX_THETA)-hi_;cn(1)=lo_-u(IDX_THETA);return cn;}
    Matrix<Scalar> cx(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
        return Matrix<Scalar>::Zero(2,x.size());}
    Matrix<Scalar> cu(const Vector<Scalar>&,const Vector<Scalar>& u)const override{
        Matrix<Scalar> J=Matrix<Scalar>::Zero(2,u.size());
        J(0,IDX_THETA)=Scalar(1);J(1,IDX_THETA)=Scalar(-1);return J;}};

// ZFloor: dz_rel >= 0  (drone at or above target altitude in relative frame)
template<typename Scalar>
class ZFloorCon:public StageConstraintBase<Scalar>{public:
    ZFloorCon(){this->constraint_type=ConstraintType::NO;this->dim_c=1;}
    Vector<Scalar> c(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
        return(Vector<Scalar>(1)<<-x(2)).finished();}
    Matrix<Scalar> cx(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
        Matrix<Scalar> J=Matrix<Scalar>::Zero(1,x.size());J(0,2)=Scalar(-1);return J;}
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override{
        return Matrix<Scalar>::Zero(1, u.size());}
};

template<typename Scalar>
class VzMinCon:public StageConstraintBase<Scalar>{Scalar vz_min_;public:
    explicit VzMinCon(double vz_min):vz_min_(vz_min){
        this->constraint_type=ConstraintType::NO;this->dim_c=1;}
    Vector<Scalar> c(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
        return(Vector<Scalar>(1)<<vz_min_-x(5)).finished();}
    Matrix<Scalar> cx(const Vector<Scalar>& x,const Vector<Scalar>&)const override{
        Matrix<Scalar> J=Matrix<Scalar>::Zero(1,x.size());J(0,5)=Scalar(-1);return J;}
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override{
        return Matrix<Scalar>::Zero(1, u.size());}
};

// SOC glideslope in relative frame. Cone apex at origin (target).
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
        cn(0) =  tan_gs_ * x(2);
        cn(1) =  x(0);
        cn(2) =  x(1);
        return -cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, NX_SS);
        J(0,2) = -tan_gs_;
        J(1,0) = Scalar(-1.0);
        J(2,1) = Scalar(-1.0);
        return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(3, u.size());
    }
};

// ── Factory ──────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& x0_rel,
    const Eigen::VectorXd& /* terminal_state */,
    const std::vector<Eigen::VectorXd>& prev_U = {},
    const std::vector<Eigen::VectorXd>& prev_X = {},
    const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero(),
    const std::vector<Eigen::MatrixXd>& prev_K = {})
{
    auto problem = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    auto dyn   = std::make_shared<Quad6DOFVarTimeRelative<double>>();
    dyn->setMass(MASS);
    dyn->setGravity(GRAVITY);
    dyn->setJb(J_B);
    dyn->setTargetAccel(target_accel);  // snapshot for this solve

    auto cost  = std::make_shared<TimeCost<double>>(1e-4, 1.0);
    auto tcost = std::make_shared<RelTermCost<double>>(500.0, 50.0);
    auto cfmin = std::make_shared<FminCon<double>>();
    auto cfmax = std::make_shared<FmaxCon<double>>();
    auto cmom  = std::make_shared<MomentCon<double>>();
    auto cth   = std::make_shared<ThetaBounds<double>>(THL, THH);
    auto czfl  = std::make_shared<ZFloorCon<double>>();
    auto cvz   = std::make_shared<VzMinCon<double>>(-VZ_LAND_MAX);
    auto cgs   = std::make_shared<GlideslopeCon_rel<double>>();

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

    Eigen::VectorXd x0ss(NX_SS); x0ss.segment(0,NX)=x0_rel; x0ss(IDX_DT)=0.0;
    problem->setInitialState(0, x0ss);

    Eigen::VectorXd sim = x0_rel;
    bool use_feedback = (!prev_U.empty() && !prev_X.empty() && !prev_K.empty()
                         && (int)prev_U.size()  >= HORIZON
                         && (int)prev_X.size()  >  HORIZON
                         && (int)prev_K.size()  >= HORIZON);

    for(int k=0; k<HORIZON; ++k){
        Eigen::VectorXd u0(NU_SS); u0.setZero();
        if (use_feedback){
            Eigen::VectorXd sim_ss(NX_SS);
            sim_ss.segment(0,NX) = sim;
            sim_ss(IDX_DT)       = static_cast<double>(k) * TH_INIT;
            Eigen::VectorXd delta = sim_ss - prev_X[k];
            u0 = prev_U[k] + prev_K[k] * delta;
            u0(0)         = std::max((double)FMIN, std::min((double)FMAX, (double)u0(0)));
            u0(1) = std::max(-TAU_MAX, std::min(TAU_MAX, (double)u0(1)));
            u0(2) = std::max(-TAU_MAX, std::min(TAU_MAX, (double)u0(2)));
            u0(3) = std::max(-TAU_MAX, std::min(TAU_MAX, (double)u0(3)));
            u0(IDX_THETA) = std::max(THL, std::min(THH, (double)u0(IDX_THETA)));
        } else if (!prev_U.empty() && k < (int)prev_U.size()){
            u0 = prev_U[k];
        } else {
            Eigen::Vector3d p0 = sim.head(3);
            Eigen::Vector3d v0 = sim.segment(3,3);
            double T_g = (HORIZON - k) * TH_INIT;
            if (T_g < TH_INIT) T_g = TH_INIT;
            Eigen::Vector3d a_req = 2.0*(-p0 - v0*T_g) / (T_g*T_g);
            const double a_req_z_min = (-VZ_LAND_MAX - v0.z()) / T_g;
            a_req.z() = std::max(a_req.z(), a_req_z_min);
            Eigen::Vector3d fw = MASS*(a_req - GRAVITY);
            double fz_g = std::max((double)FMIN, std::min((double)FMAX, fw.norm()));
            u0(0) = fz_g; u0(IDX_THETA) = TH_INIT;
        }
        problem->setInitialControl(k, u0);
        Eigen::VectorXd up = u0.segment(0,NU);
        double Th = u0(IDX_THETA);
        sim = dyn->propagate(sim, up, Th);
        sim(2) = std::max(sim(2), 0.0);
        Eigen::VectorXd xk(NX_SS); xk.segment(0,NX)=sim; xk(IDX_DT)=(k+1)*Th;
        problem->setInitialState(k+1, xk);
    }

    return problem;
}

} // namespace StateswitchOCP

