/// @file ocp_stc_landing_noaug_legacy.hpp
/// @brief ARCHIVED pre-faithful-port variant of the no-aug CT-cSTC landing arm.
///
/// This is the ORIGINAL CoManDO port as it existed before commit e02ec53
/// ("make the integrand a faithful port of the benchmark arm"), preserved
/// verbatim for A/B comparison. It differs from the benchmark
/// quad_single_horizon_noaug_stc.cpp in three ways that were inherited from
/// the ocp_stc_landing.hpp scaffold rather than chosen:
///   1. landing trigger = normalized altitude ramp * lateral-capture AND
///      ((trig_alt/ALT_TRIG) * trig_cap/cap^2) instead of raw posPart(ALT_TRIG-z)
///   2. thrust STC blocks (tight/wide) inside the CT-cSTC integrand
///   3. planning thrust bounds = actuator bounds (0.08/0.60), no margin
/// plus looser stage/tilt/omega constants (STAGE_RADIUS 0.55 etc.).
///
/// Registry keys: `stc_landing_noaug_legacy` / `rh_stc_noaug_legacy`.
/// All env tunables are prefixed SZMUK_LEGACY_* so this arm can be tuned
/// without disturbing the faithful arm in the same process.
///
/// Do not "fix" this file — its value is being an unchanged reference point.

/// @file ocp_stc_landing_noaug.hpp
/// @brief No-aug interval CT-cSTC receding-horizon landing OCP ("stc_landing_noaug").
///
/// Ported from ALIPDDP-main/problem_examples/STC/quad_single_horizon_noaug_stc.cpp
/// (the "interval_noaug" benchmark arm), grafted onto the receding-horizon
/// machinery of ocp_stc_landing.hpp (which is the terminal-eq port of
/// quad_landing_rh_clean_stc.cpp). Relative-frame landing on a moving target:
///
///   state   = [13 physical | IDX_DT cumulative time]        (NO accumulator)
///   control = [fz, Mx, My, Mz | IDX_THETA free per-step dt]
///
/// Instead of carrying the CT-cSTC violation as an augmented state y with a
/// terminal EQ y_N = 0, the SAME RK4 interval integral of the STC density H
/// is recomputed directly from (x_k, u_k) inside a per-stage inequality:
///
///   integral(H dt) over [t_k, t_k + theta_k]  <=  eps      (hard-wired, always on)
///
/// This isolates state augmentation from the constraint functional: the zero
/// set is the same, but there is no y row/column in the Riccati recursion and
/// no y defect. SZMUK_CTCS_MODE is intentionally IGNORED here (the benchmark
/// file's in-code default was a vacuous terminal_eq stub — see its header).
///
/// Tuning (noaug_basin champion): SZMUK_CTCS_STEP_EPS = 1.6e-3 with
/// SZMUK_CTCS_Y_SCALE = 10 — a ~10x looser budget and much smaller Y_SCALE
/// than the augmented arm, because Y_SCALE existed to condition the y row of
/// the augmented dynamics and there is no such row here. These are the
/// defaults below; export the SZMUK_* vars before launching to retune.
///
/// Warm-start note: the planner core's makeXshifted() does a plain index
/// shift, so create() rebases IDX_DT against the new horizon start
/// (prev_X[0] after the shift == the previously executed node), exactly as
/// ocp_stc_landing.hpp does — minus the y-rebase, which no longer exists.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <any>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
#include "planner_core/types.hpp"

namespace StcLandingNoAugLegacyOCP {

// ── Env helpers ───────────────────────────────────────────────────────────────
inline double envOrQ(const char* n, double d) {
    const char* v = std::getenv(n);
    return v ? std::atof(v) : d;
}
inline int envOrInt(const char* n, int d) {
    const char* v = std::getenv(n);
    return v ? std::atoi(v) : d;
}

// ── Dimensions ────────────────────────────────────────────────────────────────
static constexpr int NX      = 13;
static constexpr int NU      = 4;
static constexpr int NX_SS   = 14;   // NX + IDX_DT (IDX_CTCS_Y intentionally REMOVED)
static constexpr int NU_SS   = 5;    // NU + IDX_THETA
static constexpr int IDX_DT     = 13;
static constexpr int IDX_THETA  = 4;

// ── Vehicle parameters ────────────────────────────────────────────────────────
static constexpr double MASS    = 0.027;
static constexpr double IXX     = 1.66e-5;
static constexpr double IYY     = 1.66e-5;
static constexpr double IZZ     = 2.92e-5;
static constexpr double J_SCALE = 1.0 / IXX;
static constexpr double FMIN    = 0.08;
static constexpr double FMAX    = 0.6;
static constexpr double L_ARM   = 0.046;
static const double TAU_MAX = L_ARM * (FMAX/4.0 - FMIN/4.0) * J_SCALE;

static const Eigen::Matrix3d J_B = [](){
    Eigen::Matrix3d J; J.setZero();
    J(0,0)=IXX*J_SCALE; J(1,1)=IYY*J_SCALE; J(2,2)=IZZ*J_SCALE;
    return J;
}();
static const Eigen::Vector3d GRAVITY(0.0, 0.0, -9.81);

// ── Horizon / timing (env-resolved once at startup) ──────────────────────────
static const int    HORIZON = std::max(1, std::getenv("SZMUK_LEGACY_RH_N")
                                  ? std::atoi(std::getenv("SZMUK_LEGACY_RH_N"))
                                  : envOrInt("SZMUK_LEGACY_N", 30));
static const int    NEX     = std::max(1, envOrInt("SZMUK_LEGACY_RH_NEX", 7));
static const double TH0     = envOrQ("SZMUK_LEGACY_TH0", 0.07);
static const double THL     = envOrQ("SZMUK_LEGACY_THL", 0.05);
static const double THH     = envOrQ("SZMUK_LEGACY_THH", 0.12);

// ── Glideslope / floor / descent cap ─────────────────────────────────────────
static constexpr double GS_APEX_OFFSET  = 0.1;
static constexpr double VZ_LAND_MAX     = 0.5;

// ── Terminal cost weights ─────────────────────────────────────────────────────
static const double TERM_W_POS_XY  = envOrQ("SZMUK_LEGACY_TERM_W_POS_XY", 2500.0);
static const double TERM_W_POS_Z   = envOrQ("SZMUK_LEGACY_TERM_W_POS_Z", 2500.0);
static const double TERM_W_VEL_XY  = envOrQ("SZMUK_LEGACY_TERM_W_VEL_XY", 900.0);
static const double TERM_W_VEL_Z   = envOrQ("SZMUK_LEGACY_TERM_W_VEL_Z", 900.0);
static const double TERM_W_TILT    = envOrQ("SZMUK_LEGACY_TERM_W_TILT", 200.0);
static const double TERM_W_OM      = envOrQ("SZMUK_LEGACY_TERM_W_OM", 50.0);
static const double TERM_VZ_REF    = envOrQ("SZMUK_LEGACY_TERM_VZ_REF", 0.0);
static const double STAGE_W_TIME   = envOrQ("SZMUK_LEGACY_W_TIME", 0.04);
static const double R_THETA        = envOrQ("SZMUK_LEGACY_R_THETA", 10.0);
static const double THETA_REF      = envOrQ("SZMUK_LEGACY_THETA_REF", 0.10);

// ── Smoothness/control weights ────────────────────────────────────────────────
static constexpr double HOVER_THRUST = MASS * 9.81;
static const double R_CTRL_THR  = envOrQ("SZMUK_LEGACY_R_THR", 2.0);
static const double R_CTRL_MOM  = envOrQ("SZMUK_LEGACY_R_MOM", 0.6);
static const double W_TOUCH_VEL = envOrQ("SZMUK_LEGACY_W_TOUCH_VEL", 10.0);
static const double TOUCH_ALT   = envOrQ("SZMUK_LEGACY_TOUCH_ALT", 0.25);
static const double TOUCH_SCALE = envOrQ("SZMUK_LEGACY_TOUCH_SCALE", 0.25);
static const double W_STAGE_TARGET_XY  = envOrQ("SZMUK_LEGACY_W_STAGE_TARGET_XY", 0.0);
static const double W_STAGE_TARGET_Z   = envOrQ("SZMUK_LEGACY_W_STAGE_TARGET_Z", 0.0);
static const double W_STAGE_TARGET_VEL = envOrQ("SZMUK_LEGACY_W_STAGE_TARGET_VEL", 0.0);

// ── CT-cSTC constraint bounds ─────────────────────────────────────────────────
static const double THETA_PHASE0_DEG   = envOrQ("SZMUK_LEGACY_TILT_PHASE0_DEG", 15.0);
static const double OMEGA_PHASE0_MAX   = envOrQ("SZMUK_LEGACY_OMEGA_PHASE0_MAX", 0.50);
static const double SPD_PHASE0_MAX     = envOrQ("SZMUK_LEGACY_SPD_PHASE0_MAX", 3.0);
static const double THETA_STC_DEG      = envOrQ("SZMUK_LEGACY_TILT_STC_DEG", 2.0);
static const double ALPHA_THETA_STC    = envOrQ("SZMUK_LEGACY_ALPHA_THETA_STC", 1.0);
static const double OMEGA_STC_MAX      = envOrQ("SZMUK_LEGACY_OMEGA_STC_MAX", 0.05);
static const double GS_CONE_HALF_ANGLE_DEG = envOrQ("SZMUK_LEGACY_GS_CONE_DEG", 15.0);
static const double GS_CONE_TAN            = std::tan(GS_CONE_HALF_ANGLE_DEG * M_PI / 180.0);
static const double SPD_STC_MAX        = envOrQ("SZMUK_LEGACY_SPD_STC_MAX", 0.75);
static const double W_LAND_SPEED       = envOrQ("SZMUK_LEGACY_W_LAND_SPEED", 1.0);
static const double W_LAND_TILT        = envOrQ("SZMUK_LEGACY_W_LAND_TILT", 1.0e5);
static const double W_LAND_OMEGA       = envOrQ("SZMUK_LEGACY_W_LAND_OMEGA", 100.0);
static const double W_LAND_GS          = envOrQ("SZMUK_LEGACY_W_LAND_GS", 5000.0);
static const double LOS_CONE_HALF_ANGLE_DEG = envOrQ("SZMUK_LEGACY_LOS_CONE_DEG", 30.0);
static const double LOS_CONE_TAN            = std::tan(LOS_CONE_HALF_ANGLE_DEG * M_PI / 180.0);
static const double W_STC_LOS               = envOrQ("SZMUK_LEGACY_W_STC_LOS", 50.0);
static const double LOS_ALT_TRIG            = envOrQ("SZMUK_LEGACY_LOS_ALT_TRIG", 1.8);
static const double LOS_TRIGGER_SCALE       = envOrQ("SZMUK_LEGACY_LOS_TRIGGER_SCALE", 0.25);
static const double LOS_TRIGGER_FLOOR       = envOrQ("SZMUK_LEGACY_LOS_TRIGGER_FLOOR", 0.0);
static constexpr double T_MIN_AFT          = 0.21;
static constexpr double T_MAX_AFT          = 0.40;
static constexpr double T_MIN_WIDE         = 0.12;
static constexpr double T_MAX_WIDE         = 0.52;
static constexpr double V_THRUST_TRIG          = 1.0;
static constexpr double THETA_THRUST_TRIG_DEG  = 15.0;

// ── Altitude trigger / lateral capture cost gate ─────────────────────────────
static const double ALT_TRIG        = envOrQ("SZMUK_LEGACY_ALT_TRIG", 0.8);
static constexpr double CAP_COST_RADIUS = 0.50;
static const double CAP_LAND_RADIUS = envOrQ("SZMUK_LEGACY_RH_CAP_LAND_RADIUS", 0.45);
static constexpr double K_CAP          = 20.0;
static const double Z_STAGE            = envOrQ("SZMUK_LEGACY_Z_STAGE", 1.8);
static const double STAGE_RADIUS       = envOrQ("SZMUK_LEGACY_STAGE_RADIUS", 0.55);
static const double STAGE_TRIGGER_SCALE = envOrQ("SZMUK_LEGACY_STAGE_TRIGGER_SCALE", 0.11);
static const double STAGE_TRIGGER_FLOOR = envOrQ("SZMUK_LEGACY_STAGE_TRIGGER_FLOOR", 0.0);
static const double STAGE_SETTLE_TILT_DEG  = envOrQ("SZMUK_LEGACY_STAGE_SETTLE_TILT_DEG", 12.0);
static const double STAGE_SETTLE_OMEGA_MAX = envOrQ("SZMUK_LEGACY_STAGE_SETTLE_OMEGA_MAX", 0.50);
static const double W_STC_STAGE        = envOrQ("SZMUK_LEGACY_W_STC_STAGE", 10.0);
static constexpr double CAP_DESC_RADIUS = 0.12;
static constexpr double K_DESC          = 40.0;

// ── Runtime toggles ───────────────────────────────────────────────────────────
inline bool useTriggeredLandingStc() {
    const char* env = std::getenv("SZMUK_LEGACY_QUAD_TRIG_STC");
    return !(env && std::string(env) == "0");
}

inline std::string stageTriggerMode() {
    const char* env = std::getenv("SZMUK_LEGACY_STAGE_TRIGGER");
    return env ? std::string(env) : std::string("radius");
}

// NOTE: no ctcsEnforcementMode() here — the interval inequality is hard-wired
// (SZMUK_CTCS_MODE is deliberately ignored; see file header).

inline bool useRhFeedbackWarmStart() {
    const char* env = std::getenv("SZMUK_LEGACY_RH_FEEDBACK_WARM_START");
    return !(env && std::string(env) == "0");
}

inline bool useRhAltOnlyLandingTrigger() {
    const char* env = std::getenv("SZMUK_LEGACY_RH_ALT_ONLY_LANDING_TRIGGER");
    return env && std::string(env) != "0";
}

// Density rescaling (rocket Y_SCALE pattern, kept for tuning parity with the
// benchmark arm): the constraint sees integral(H/Y_SCALE dt) <= EPS/Y_SCALE,
// so the zero set is unchanged; there is no y row to condition here, but the
// scaled units keep the noaug_basin (eps, yscale) tuning table applicable.
// noaug_basin champion: EPS=1.6e-3, Y_SCALE=10.
static const double CTCS_Y_SCALE = [] {
    const double s = envOrQ("SZMUK_LEGACY_CTCS_Y_SCALE", 10.0);
    return s > 0.0 ? s : 1.0;
}();
static const double CTCS_STEP_EPS = envOrQ("SZMUK_LEGACY_CTCS_STEP_EPS", 1.6e-3);

// ── Sigmoid helpers ───────────────────────────────────────────────────────────
template<typename Scalar>
inline Scalar sigmaLandVal(const Scalar& rxy2) {
    const Scalar g = Scalar(CAP_COST_RADIUS * CAP_COST_RADIUS) - rxy2;
    return Scalar(0.5) * (Scalar(1) + std::tanh(Scalar(K_CAP) * g));
}

template<typename Scalar>
inline void sigmaLandGrad(const Scalar& rxy2, const Scalar& x0, const Scalar& x1,
                          Scalar& sig, Scalar& gx0, Scalar& gx1) {
    const Scalar g       = Scalar(CAP_COST_RADIUS * CAP_COST_RADIUS) - rxy2;
    const Scalar th      = std::tanh(Scalar(K_CAP) * g);
    sig     = Scalar(0.5) * (Scalar(1) + th);
    const Scalar dsig_dg = Scalar(0.5) * Scalar(K_CAP) * (Scalar(1) - th * th);
    gx0 = dsig_dg * Scalar(-2) * x0;
    gx1 = dsig_dg * Scalar(-2) * x1;
}

template<typename Scalar>
inline Scalar sigmaDescVal(const Scalar& rxy2, const Scalar& /*z*/) {
    const Scalar g = Scalar(CAP_DESC_RADIUS * CAP_DESC_RADIUS) - rxy2;
    return Scalar(0.5) * (Scalar(1) + std::tanh(Scalar(K_DESC) * g));
}

template<typename Scalar>
inline void sigmaDescGrad(const Scalar& rxy2, const Scalar& x0, const Scalar& x1,
                          const Scalar& /*z*/,
                          Scalar& sig, Scalar& gx0, Scalar& gx1, Scalar& gz) {
    const Scalar g    = Scalar(CAP_DESC_RADIUS * CAP_DESC_RADIUS) - rxy2;
    const Scalar th   = std::tanh(Scalar(K_DESC) * g);
    sig = Scalar(0.5) * (Scalar(1) + th);
    const Scalar dsig = Scalar(0.5) * Scalar(K_DESC) * (Scalar(1) - th * th);
    gx0 = dsig * Scalar(-2) * x0;
    gx1 = dsig * Scalar(-2) * x1;
    gz  = Scalar(0);
}

template<typename Scalar>
inline Scalar posPart(const Scalar& v) {
    return (v > Scalar(0)) ? v : Scalar(0);
}

template<typename Scalar>
inline Scalar landingAltitudeTrigger(const Scalar& z, const Scalar& rxy2) {
    const Scalar trig_alt = posPart(Scalar(ALT_TRIG) - z);
    if (useRhAltOnlyLandingTrigger()) return trig_alt;
    const Scalar cap2 = Scalar(CAP_LAND_RADIUS * CAP_LAND_RADIUS);
    const Scalar trig_cap = posPart(cap2 - rxy2) / cap2;
    return (trig_alt / Scalar(ALT_TRIG)) * trig_cap;
}

inline double computeSigmaLand(const Eigen::VectorXd& x) {
    const double rxy2 = x(0)*x(0) + x(1)*x(1);
    return sigmaLandVal<double>(rxy2);
}

inline double computeTrigAlt(const Eigen::VectorXd& x) {
    const double rxy2 = x(0)*x(0) + x(1)*x(1);
    return landingAltitudeTrigger<double>(x(2), rxy2);
}

// ── Constant-accel target predictor ───────────────────────────────────────────
// Reset from the TargetSnapshot at every solve (see prepare_extra), then held
// as a constant-acceleration model across the horizon.
struct ConstantAccelPredictor {
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d a = Eigen::Vector3d::Zero();

    void setState(const Eigen::Vector3d& p_, const Eigen::Vector3d& v_,
                  const Eigen::Vector3d& a_) {
        p = p_; v = v_; a = a_;
    }

    Eigen::Vector3d predictPos(double dt)   const { return p + v*dt + 0.5*a*dt*dt; }
    Eigen::Vector3d predictVel(double dt)   const { return v + a*dt; }
    Eigen::Vector3d predictAccel(double dt) const { (void)dt; return a; }
};

struct StcLandingNoAugLegacyExtra {
    std::shared_ptr<ConstantAccelPredictor> predictor;
};

template<typename Scalar>
Scalar evalCtcsIntegrand(const Vector<Scalar>& x,
                         const Vector<Scalar>& u);

// ── Relative dynamics with constant-accel target prediction (NO y-row) ───────
template<typename Scalar>
class Quad6DOFVarTimeRelativePred : public Quad6DOFVarTimeRelative<Scalar> {
    std::shared_ptr<ConstantAccelPredictor> pred_;

public:
    explicit Quad6DOFVarTimeRelativePred(std::shared_ptr<ConstantAccelPredictor> pred)
        : Quad6DOFVarTimeRelative<Scalar>(), pred_(std::move(pred)) {
        this->dim_x = NX_SS;
    }

private:
    double ctcsRate(const Eigen::VectorXd& x_phys, const Eigen::VectorXd& u_full) const {
        Eigen::VectorXd x_aug(NX_SS);
        x_aug.setZero();
        x_aug.segment(0, this->NX_PHYS) = x_phys;
        return evalCtcsIntegrand<double>(x_aug, u_full) / CTCS_Y_SCALE;
    }

public:
    // The SAME RK4 interval integral the accumulator would have added to y,
    // computed directly from (x,u). No augmented state is involved.
    double intervalIntegral(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        auto xd = x.segment(0, this->NX_PHYS).template cast<double>();
        auto ud = u.segment(0, this->NU_PHYS).template cast<double>();
        auto u_full = u.template cast<double>();
        double Th   = static_cast<double>(u(IDX_THETA));
        double tabs = static_cast<double>(x(IDX_DT));

        Eigen::Vector3d a1 = pred_->predictAccel(tabs);
        auto k1 = this->xdot_impl(xd, ud, a1);
        double h1 = ctcsRate(xd, u_full);
        Eigen::Vector3d a2 = pred_->predictAccel(tabs + 0.5*Th);
        Eigen::VectorXd x2 = xd + 0.5*Th*k1;
        auto k2 = this->xdot_impl(x2, ud, a2);
        double h2 = ctcsRate(x2, u_full);
        Eigen::Vector3d a3 = pred_->predictAccel(tabs + 0.5*Th);
        Eigen::VectorXd x3 = xd + 0.5*Th*k2;
        auto k3 = this->xdot_impl(x3, ud, a3);
        double h3 = ctcsRate(x3, u_full);
        Eigen::Vector3d a4 = pred_->predictAccel(tabs + Th);
        Eigen::VectorXd x4 = xd + Th*k3;
        (void)a4;
        double h4 = ctcsRate(x4, u_full);
        return (Th/6.0)*(h1 + 2.0*h2 + 2.0*h3 + h4);
    }

    Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto xd = x.segment(0, this->NX_PHYS).template cast<double>();
        auto ud = u.segment(0, this->NU_PHYS).template cast<double>();
        double Th   = static_cast<double>(u(IDX_THETA));
        double tabs = static_cast<double>(x(IDX_DT));

        Eigen::Vector3d a1 = pred_->predictAccel(tabs);
        auto k1 = this->xdot_impl(xd, ud, a1);

        Eigen::Vector3d a2 = pred_->predictAccel(tabs + 0.5*Th);
        Eigen::VectorXd x2 = xd + 0.5*Th*k1;
        auto k2 = this->xdot_impl(x2, ud, a2);

        Eigen::Vector3d a3 = pred_->predictAccel(tabs + 0.5*Th);
        Eigen::VectorXd x3 = xd + 0.5*Th*k2;
        auto k3 = this->xdot_impl(x3, ud, a3);

        Eigen::Vector3d a4 = pred_->predictAccel(tabs + Th);
        Eigen::VectorXd x4 = xd + Th*k3;
        auto k4 = this->xdot_impl(x4, ud, a4);

        Eigen::VectorXd xn = xd + (Th/6.0)*(k1 + 2*k2 + 2*k3 + k4);
        xn.segment(6, 4).normalize();

        const_cast<Quad6DOFVarTimeRelativePred*>(this)
            ->setTargetAccel(pred_->predictAccel(tabs + 0.5*Th));

        Vector<Scalar> res(NX_SS);
        res.setZero();
        res.segment(0, this->NX_PHYS) = xn.template cast<Scalar>();
        res(IDX_DT) = x(IDX_DT) + u(IDX_THETA);
        return res;
    }

    Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto xd = x.segment(0, this->NX_PHYS).template cast<double>();
        auto ud = u.segment(0, this->NU_PHYS).template cast<double>();
        const double Th   = static_cast<double>(u(IDX_THETA));
        const double tabs = static_cast<double>(x(IDX_DT));
        const Eigen::Vector3d amid = pred_->predictAccel(tabs + 0.5 * Th);
        auto j = this->rk4_jac_impl(xd, ud, Th, amid);

        Matrix<Scalar> Fx = Matrix<Scalar>::Zero(NX_SS, NX_SS);
        Fx.block(0, 0, this->NX_PHYS, this->NX_PHYS) = j.dxdx.template cast<Scalar>();
        Fx(IDX_DT, IDX_DT) = Scalar(1.0);
        const_cast<Quad6DOFVarTimeRelativePred*>(this)->setTargetAccel(amid);
        return Fx;
    }

    Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto xd = x.segment(0, this->NX_PHYS).template cast<double>();
        auto ud = u.segment(0, this->NU_PHYS).template cast<double>();
        const double Th   = static_cast<double>(u(IDX_THETA));
        const double tabs = static_cast<double>(x(IDX_DT));
        const Eigen::Vector3d amid = pred_->predictAccel(tabs + 0.5 * Th);
        auto j = this->rk4_jac_impl(xd, ud, Th, amid);

        Matrix<Scalar> Fu = Matrix<Scalar>::Zero(NX_SS, NU_SS);
        Fu.block(0, 0, this->NX_PHYS, this->NU_PHYS) = j.dxdu.template cast<Scalar>();
        Fu.block(0, IDX_THETA, this->NX_PHYS, 1) = j.dxdT.template cast<Scalar>();
        Fu(IDX_DT, IDX_THETA) = Scalar(1.0);
        const_cast<Quad6DOFVarTimeRelativePred*>(this)->setTargetAccel(amid);
        return Fu;
    }

    // 14-dim propagate for warm-start rollouts (tracks IDX_DT).
    Eigen::VectorXd propagate14(const Eigen::VectorXd& x14,
                                const Eigen::VectorXd& u_phys,
                                double Th) const {
        double tabs = x14(IDX_DT);
        auto xd = x14.segment(0, this->NX_PHYS);

        Eigen::Vector3d a1 = pred_->predictAccel(tabs);
        auto k1 = this->xdot_impl(xd, u_phys, a1);
        Eigen::Vector3d a2 = pred_->predictAccel(tabs + 0.5*Th);
        Eigen::VectorXd x2 = xd + 0.5*Th*k1;
        auto k2 = this->xdot_impl(x2, u_phys, a2);
        Eigen::Vector3d a3 = pred_->predictAccel(tabs + 0.5*Th);
        Eigen::VectorXd x3 = xd + 0.5*Th*k2;
        auto k3 = this->xdot_impl(x3, u_phys, a3);
        Eigen::Vector3d a4 = pred_->predictAccel(tabs + Th);
        Eigen::VectorXd x4 = xd + Th*k3;
        auto k4 = this->xdot_impl(x4, u_phys, a4);

        Eigen::VectorXd xn14(NX_SS);
        xn14.setZero();
        xn14.segment(0, this->NX_PHYS) = xd + (Th/6.0)*(k1+2*k2+2*k3+k4);
        xn14.segment(6, 4).normalize();
        xn14(IDX_DT) = x14(IDX_DT) + Th;
        return xn14;
    }
};

// ── CT-cSTC integrand ─────────────────────────────────────────────────────────
template<typename Scalar>
struct CtcsIntegrandTerms {
    Scalar a_stage            = Scalar(0);
    Scalar a_los              = Scalar(0);
    Scalar a_altitude_state   = Scalar(0);
    Scalar a_thrust           = Scalar(0);
    Scalar loose_tilt         = Scalar(0);
    Scalar loose_omega        = Scalar(0);
    Scalar speed              = Scalar(0);
    Scalar tilt               = Scalar(0);
    Scalar omega              = Scalar(0);
    Scalar glideslope         = Scalar(0);
    Scalar los                = Scalar(0);
    Scalar thrust_hi          = Scalar(0);
    Scalar thrust_lo          = Scalar(0);
    Scalar thrust_wide_hi     = Scalar(0);
    Scalar thrust_wide_lo     = Scalar(0);
    Scalar path_value         = Scalar(0);
    Scalar staging_value      = Scalar(0);
    Scalar los_value          = Scalar(0);
    Scalar landing_value      = Scalar(0);
    Scalar thrust_tight_value = Scalar(0);
    Scalar thrust_wide_value  = Scalar(0);
    Scalar stc_value          = Scalar(0);
    Scalar value              = Scalar(0);
};

template<typename Scalar>
inline CtcsIntegrandTerms<Scalar> evalCtcsIntegrandTerms(const Vector<Scalar>& x,
                                                         const Vector<Scalar>& u) {
    CtcsIntegrandTerms<Scalar> out;

    const Scalar rxy2     = x(0)*x(0) + x(1)*x(1);
    out.a_altitude_state  = landingAltitudeTrigger<Scalar>(x(2), rxy2);

    const Scalar v2     = x.template segment<3>(3).squaredNorm();
    const Scalar qperp2 = x(7)*x(7) + x(8)*x(8);
    const Scalar om2    = x.template segment<3>(10).squaredNorm();
    const Scalar rxy    = std::sqrt(rxy2 + Scalar(1e-12));
    const Scalar v_spd  = std::sqrt(v2   + Scalar(1e-12));
    const Scalar stage_radius_margin =
        (rxy - Scalar(STAGE_RADIUS)) / Scalar(STAGE_TRIGGER_SCALE);
    const Scalar stage_radius_trig =
        (stage_radius_margin > Scalar(0))
            ? Scalar(STAGE_TRIGGER_FLOOR) + stage_radius_margin
            : Scalar(0);
    const Scalar bounded_stage_trig = Scalar(1) - sigmaLandVal(rxy2);
    const Scalar stage_settle_tilt_lim =
        Scalar(std::pow(std::sin(0.5 * STAGE_SETTLE_TILT_DEG * M_PI / 180.0), 2));
    const Scalar stage_settle_tilt = posPart(qperp2 - stage_settle_tilt_lim);
    const Scalar stage_settle_omega =
        posPart(om2 - Scalar(STAGE_SETTLE_OMEGA_MAX * STAGE_SETTLE_OMEGA_MAX));
    const std::string stage_mode = stageTriggerMode();
    if (stage_mode == "settle") {
        out.a_stage = stage_settle_tilt + stage_settle_omega;
    } else if (stage_mode == "radius") {
        out.a_stage = stage_radius_trig;
    } else {
        out.a_stage = bounded_stage_trig;
    }

    const Scalar loose_tilt_lim =
        Scalar(std::pow(std::sin(0.5 * THETA_PHASE0_DEG * M_PI / 180.0), 2));
    out.loose_tilt  = posPart(qperp2 - loose_tilt_lim);
    out.loose_omega = posPart(om2 - Scalar(OMEGA_PHASE0_MAX * OMEGA_PHASE0_MAX));

    const Scalar tilt_lim =
        Scalar(std::pow(std::sin(0.5 * ALPHA_THETA_STC * THETA_STC_DEG * M_PI / 180.0), 2));
    out.speed      = posPart(v2     - Scalar(SPD_STC_MAX   * SPD_STC_MAX));
    out.tilt       = posPart(qperp2 - tilt_lim);
    out.omega      = posPart(om2    - Scalar(OMEGA_STC_MAX * OMEGA_STC_MAX));
    const Scalar gs_val = rxy - Scalar(GS_CONE_TAN) * (x(2) + Scalar(GS_APEX_OFFSET));
    out.glideslope = posPart(gs_val);

    const Scalar qw = x(6), qx = x(7), qy = x(8), qz = x(9);
    const Scalar bz_x = Scalar(2) * (qx * qz + qw * qy);
    const Scalar bz_y = Scalar(2) * (qy * qz - qw * qx);
    const Scalar bz_z = qw*qw - qx*qx - qy*qy + qz*qz;
    const Scalar cam_x = -bz_x;
    const Scalar cam_y = -bz_y;
    const Scalar cam_z = -bz_z;
    const Scalar los_x = -x(0);
    const Scalar los_y = -x(1);
    const Scalar los_z = -x(2);
    const Scalar axial = los_x * cam_x + los_y * cam_y + los_z * cam_z;
    const Scalar los_norm2 = los_x*los_x + los_y*los_y + los_z*los_z;
    const Scalar lat2 = posPart(los_norm2 - axial * axial);
    const Scalar lateral = std::sqrt(lat2 + Scalar(1e-12));
    const Scalar los_alt_margin =
        (Scalar(LOS_ALT_TRIG) - x(2)) / Scalar(LOS_TRIGGER_SCALE);
    out.a_los =
        (los_alt_margin > Scalar(0))
            ? Scalar(LOS_TRIGGER_FLOOR) + los_alt_margin
            : Scalar(0);
    out.los = posPart(lateral - Scalar(LOS_CONE_TAN) * axial);

    const Scalar tilt_trig_thresh =
        Scalar(std::pow(std::sin(0.5 * THETA_THRUST_TRIG_DEG * M_PI / 180.0), 2));

    const Scalar trig_slow    = posPart(Scalar(V_THRUST_TRIG) - v_spd);
    const Scalar trig_upright = posPart(tilt_trig_thresh - qperp2);
    out.a_thrust  = out.a_altitude_state * trig_slow * trig_upright;
    out.thrust_hi = posPart(u(0) - Scalar(T_MAX_AFT));
    out.thrust_lo = posPart(Scalar(T_MIN_AFT) - u(0));

    const Scalar trig_fast        = posPart(v_spd  - Scalar(V_THRUST_TRIG));
    const Scalar trig_tilted      = posPart(qperp2 - tilt_trig_thresh);
    const Scalar wide_trigger_sum = trig_fast * trig_fast + trig_tilted * trig_tilted;
    out.thrust_wide_hi = posPart(u(0) - Scalar(T_MAX_WIDE));
    out.thrust_wide_lo = posPart(Scalar(T_MIN_WIDE) - u(0));

    const Scalar path_sum =
        out.loose_tilt  * out.loose_tilt  +
        out.loose_omega * out.loose_omega;

    const Scalar landing_sum =
        Scalar(W_LAND_SPEED) * out.speed      * out.speed      +
        Scalar(W_LAND_TILT)  * out.tilt       * out.tilt       +
        Scalar(W_LAND_OMEGA) * out.omega      * out.omega      +
        Scalar(W_LAND_GS)    * out.glideslope * out.glideslope;

    const Scalar thrust_tight_sum =
        out.thrust_hi * out.thrust_hi +
        out.thrust_lo * out.thrust_lo;

    const Scalar thrust_wide_sum =
        out.thrust_wide_hi * out.thrust_wide_hi +
        out.thrust_wide_lo * out.thrust_wide_lo;

    out.path_value = path_sum;

    const bool trig_stc_on = useTriggeredLandingStc();

    if (trig_stc_on) {
        const Scalar stage_height_viol = posPart(Scalar(Z_STAGE) - x(2));
        out.staging_value = Scalar(W_STC_STAGE) *
                            out.a_stage * out.a_stage *
                            stage_height_viol * stage_height_viol;

        out.los_value = Scalar(W_STC_LOS) *
                        out.a_los * out.a_los *
                        out.los * out.los;

        out.landing_value =
            out.a_altitude_state * out.a_altitude_state * landing_sum;

        out.thrust_tight_value =
            out.a_thrust * out.a_thrust * thrust_tight_sum;

        out.thrust_wide_value =
            out.a_altitude_state * out.a_altitude_state *
            wide_trigger_sum * thrust_wide_sum;
    }

    out.stc_value =
        out.staging_value      +
        out.los_value          +
        out.landing_value      +
        out.thrust_tight_value +
        out.thrust_wide_value;

    out.value =
        out.path_value +
        out.stc_value;

    return out;
}

template<typename Scalar>
Scalar evalCtcsIntegrand(const Vector<Scalar>& x,
                         const Vector<Scalar>& u) {
    return evalCtcsIntegrandTerms(x, u).value;
}

// ── Stage cost ────────────────────────────────────────────────────────────────
template<typename Scalar>
class CTcSTCStageCost : public StageCostBase<Scalar> {
    Scalar eps_;
    Scalar w_pos_xy_, w_pos_z_, w_vel_;
    Scalar rf_x_, rf_y_, rf_z_;

public:
    CTcSTCStageCost(double eps, double w_pos_xy, double w_pos_z, double w_vel,
                    const Eigen::Vector3d& r_follow)
        : eps_(Scalar(eps)),
          w_pos_xy_(Scalar(w_pos_xy)), w_pos_z_(Scalar(w_pos_z)),
          w_vel_(Scalar(w_vel)),
          rf_x_(Scalar(r_follow(0))), rf_y_(Scalar(r_follow(1))), rf_z_(Scalar(r_follow(2))) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        const Scalar rxy2 = x(0)*x(0) + x(1)*x(1);
        const Scalar sc   = sigmaLandVal(rxy2);
        const Scalar sd   = sigmaDescVal(rxy2, x(2));

        const Scalar rref_x = (Scalar(1) - sc) * rf_x_;
        const Scalar rref_y = (Scalar(1) - sc) * rf_y_;
        const Scalar rref_z = (Scalar(1) - sc) * rf_z_ + sc*(Scalar(1)-sd)*Scalar(Z_STAGE);

        const Scalar drx = x(0) - rref_x;
        const Scalar dry = x(1) - rref_y;
        const Scalar drz = x(2) - rref_z;
        const Scalar v2  = x.template segment<3>(3).squaredNorm();
        const Scalar touch_margin = (Scalar(TOUCH_ALT) - x(2)) / Scalar(TOUCH_SCALE);
        const Scalar touch = touch_margin > Scalar(0) ? touch_margin : Scalar(0);

        const Scalar pos_cost = w_pos_xy_*(drx*drx + dry*dry) + w_pos_z_*drz*drz;
        const Scalar vel_cost = sd * w_vel_ * v2;
        const Scalar touch_vel_cost = Scalar(W_TOUCH_VEL) * touch * touch * v2;
        const Scalar reg      = eps_*(v2 + x.template segment<3>(10).squaredNorm());

        const Scalar dT  = u(0) - Scalar(HOVER_THRUST);
        const Scalar ctrl_cost = Scalar(R_CTRL_THR)*dT*dT
                               + Scalar(R_CTRL_MOM)*(u(1)*u(1) + u(2)*u(2) + u(3)*u(3));

        const Scalar dTheta = u(IDX_THETA) - Scalar(THETA_REF);
        const Scalar time_reg = Scalar(R_THETA) * dTheta * dTheta;

        return Scalar(STAGE_W_TIME)*u(IDX_THETA) + time_reg + reg
             + pos_cost + vel_cost + touch_vel_cost + ctrl_cost;
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        const Scalar rxy2 = x(0)*x(0) + x(1)*x(1);

        Scalar sc, sc_gx0, sc_gx1;
        sigmaLandGrad(rxy2, x(0), x(1), sc, sc_gx0, sc_gx1);

        Scalar sd, sd_gx0, sd_gx1, sd_gz;
        sigmaDescGrad(rxy2, x(0), x(1), x(2), sd, sd_gx0, sd_gx1, sd_gz);

        const Scalar Zs    = Scalar(Z_STAGE);
        const Scalar rref_x = (Scalar(1) - sc) * rf_x_;
        const Scalar rref_y = (Scalar(1) - sc) * rf_y_;
        const Scalar rref_z = (Scalar(1) - sc) * rf_z_ + sc*(Scalar(1)-sd)*Zs;

        const Scalar drx = x(0) - rref_x;
        const Scalar dry = x(1) - rref_y;
        const Scalar drz = x(2) - rref_z;
        const Scalar v2  = x.template segment<3>(3).squaredNorm();
        const Scalar touch_margin = (Scalar(TOUCH_ALT) - x(2)) / Scalar(TOUCH_SCALE);
        const Scalar touch = touch_margin > Scalar(0) ? touch_margin : Scalar(0);
        const Scalar dtouch_dz = touch_margin > Scalar(0) ? -Scalar(1) / Scalar(TOUCH_SCALE) : Scalar(0);

        const Scalar ddrx_dx0 = Scalar(1) + sc_gx0 * rf_x_;
        const Scalar ddrx_dx1 = sc_gx1 * rf_x_;
        const Scalar ddry_dx0 = sc_gx0 * rf_y_;
        const Scalar ddry_dx1 = Scalar(1) + sc_gx1 * rf_y_;

        const Scalar ddrz_dx0 = sc_gx0*(rf_z_ - Zs*(Scalar(1)-sd)) + sc*sd_gx0*Zs;
        const Scalar ddrz_dx1 = sc_gx1*(rf_z_ - Zs*(Scalar(1)-sd)) + sc*sd_gx1*Zs;
        const Scalar ddrz_dz  = Scalar(1) + sc * sd_gz * Zs;

        Vector<Scalar> g = Vector<Scalar>::Zero(NX_SS);

        g(0) = Scalar(2)*w_pos_xy_*(drx*ddrx_dx0 + dry*ddry_dx0)
             + Scalar(2)*w_pos_z_ * drz * ddrz_dx0;
        g(1) = Scalar(2)*w_pos_xy_*(drx*ddrx_dx1 + dry*ddry_dx1)
             + Scalar(2)*w_pos_z_ * drz * ddrz_dx1;
        g(2) = Scalar(2)*w_pos_z_ * drz * ddrz_dz;

        g(0) += sd_gx0 * w_vel_ * v2;
        g(1) += sd_gx1 * w_vel_ * v2;
        g(2) += sd_gz  * w_vel_ * v2;
        g.template segment<3>(3) =
            (Scalar(2)*eps_ + Scalar(2)*sd*w_vel_ + Scalar(2)*Scalar(W_TOUCH_VEL)*touch*touch)
            * x.template segment<3>(3);
        g(2) += Scalar(2)*Scalar(W_TOUCH_VEL)*touch*dtouch_dz*v2;
        g.template segment<3>(10) = Scalar(2)*eps_ * x.template segment<3>(10);

        return g;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> g = Vector<Scalar>::Zero(NU_SS);
        g(IDX_THETA) = Scalar(STAGE_W_TIME) + Scalar(2 * R_THETA) * (u(IDX_THETA) - Scalar(THETA_REF));
        g(0) = Scalar(2 * R_CTRL_THR) * (u(0) - Scalar(HOVER_THRUST));
        g(1) = Scalar(2 * R_CTRL_MOM) * u(1);
        g(2) = Scalar(2 * R_CTRL_MOM) * u(2);
        g(3) = Scalar(2 * R_CTRL_MOM) * u(3);
        return g;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        const Scalar rxy2 = x(0)*x(0) + x(1)*x(1);

        Scalar sc, sc_gx0, sc_gx1;
        sigmaLandGrad(rxy2, x(0), x(1), sc, sc_gx0, sc_gx1);
        Scalar sd, sd_gx0, sd_gx1, sd_gz;
        sigmaDescGrad(rxy2, x(0), x(1), x(2), sd, sd_gx0, sd_gx1, sd_gz);
        const Scalar Zs = Scalar(Z_STAGE);
        const Scalar touch_margin = (Scalar(TOUCH_ALT) - x(2)) / Scalar(TOUCH_SCALE);
        const Scalar touch = touch_margin > Scalar(0) ? touch_margin : Scalar(0);
        const Scalar dtouch_dz = touch_margin > Scalar(0) ? -Scalar(1) / Scalar(TOUCH_SCALE) : Scalar(0);
        const Scalar v2  = x.template segment<3>(3).squaredNorm();

        const Scalar ddrx_dx0 = Scalar(1) + sc_gx0 * rf_x_;
        const Scalar ddrx_dx1 = sc_gx1 * rf_x_;
        const Scalar ddry_dx0 = sc_gx0 * rf_y_;
        const Scalar ddry_dx1 = Scalar(1) + sc_gx1 * rf_y_;
        const Scalar ddrz_dx0 = sc_gx0*(rf_z_ - Zs*(Scalar(1)-sd)) + sc*sd_gx0*Zs;
        const Scalar ddrz_dx1 = sc_gx1*(rf_z_ - Zs*(Scalar(1)-sd)) + sc*sd_gx1*Zs;
        const Scalar ddrz_dz  = Scalar(1) + sc * sd_gz * Zs;

        Matrix<Scalar> H = Matrix<Scalar>::Zero(NX_SS, NX_SS);

        H(0,0) = Scalar(2)*w_pos_xy_*(ddrx_dx0*ddrx_dx0 + ddry_dx0*ddry_dx0)
               + Scalar(2)*w_pos_z_*ddrz_dx0*ddrz_dx0;
        H(1,1) = Scalar(2)*w_pos_xy_*(ddrx_dx1*ddrx_dx1 + ddry_dx1*ddry_dx1)
               + Scalar(2)*w_pos_z_*ddrz_dx1*ddrz_dx1;
        H(2,2) = Scalar(2)*w_pos_z_*ddrz_dz*ddrz_dz;
        H(0,1) = H(1,0) = Scalar(2)*w_pos_xy_*(ddrx_dx0*ddrx_dx1 + ddry_dx0*ddry_dx1)
                         + Scalar(2)*w_pos_z_*ddrz_dx0*ddrz_dx1;
        H(0,2) = H(2,0) = Scalar(2)*w_pos_z_*ddrz_dx0*ddrz_dz;
        H(1,2) = H(2,1) = Scalar(2)*w_pos_z_*ddrz_dx1*ddrz_dz;

        for (int i = 3;  i < 6;  ++i) {
            H(i,i) = Scalar(2)*eps_ + Scalar(2)*sd*w_vel_
                   + Scalar(2)*Scalar(W_TOUCH_VEL)*touch*touch;
            H(2,i) += Scalar(4)*Scalar(W_TOUCH_VEL)*touch*dtouch_dz*x(i);
            H(i,2) += Scalar(4)*Scalar(W_TOUCH_VEL)*touch*dtouch_dz*x(i);
        }
        H(2,2) += Scalar(2)*Scalar(W_TOUCH_VEL)*dtouch_dz*dtouch_dz*v2;
        for (int i = 10; i < 13; ++i) H(i,i) = Scalar(2)*eps_;
        return H;
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(NU_SS, NU_SS);
        H(0, 0) = Scalar(2 * R_CTRL_THR);
        H(1, 1) = Scalar(2 * R_CTRL_MOM);
        H(2, 2) = Scalar(2 * R_CTRL_MOM);
        H(3, 3) = Scalar(2 * R_CTRL_MOM);
        H(IDX_THETA, IDX_THETA) = Scalar(2 * R_THETA);
        return H;
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(NX_SS, NU_SS);
    }
};

// ── Terminal cost ─────────────────────────────────────────────────────────────
template<typename Scalar>
class RelTermCost : public TerminalCostBase<Scalar> {
    double wp_xy_, wp_z_, wv_xy_, wv_z_, watt_, wom_, vz_ref_;
public:
    explicit RelTermCost(double wp_xy=500.0, double wp_z=500.0,
                         double wv_xy=50.0, double wv_z=100.0,
                         double watt=200.0, double wom=50.0, double vz_ref=-0.1)
        : wp_xy_(wp_xy), wp_z_(wp_z), wv_xy_(wv_xy), wv_z_(wv_z),
          watt_(watt), wom_(wom), vz_ref_(vz_ref) {}

    static void bodyZHorizontal(const Vector<Scalar>& x, Scalar& bx, Scalar& by) {
        const Scalar qw = x(6), qx = x(7), qy = x(8), qz = x(9);
        bx = Scalar(2) * (qx * qz + qw * qy);
        by = Scalar(2) * (qy * qz - qw * qx);
    }

    static Matrix<Scalar> bodyZHorizontalJacobian(const Vector<Scalar>& x) {
        const Scalar qw = x(6), qx = x(7), qy = x(8), qz = x(9);
        Matrix<Scalar> J = Matrix<Scalar>::Zero(2, 4);
        J(0, 0) = Scalar(2) * qy;  J(0, 1) = Scalar(2) * qz;
        J(0, 2) = Scalar(2) * qw;  J(0, 3) = Scalar(2) * qx;
        J(1, 0) = Scalar(-2) * qx; J(1, 1) = Scalar(-2) * qw;
        J(1, 2) = Scalar(2) * qz;  J(1, 3) = Scalar(2) * qy;
        return J;
    }

    Scalar p(const Vector<Scalar>& x) const override {
        Scalar ep   = x.template segment<2>(0).squaredNorm();
        Scalar ez   = x(2)*x(2);
        Scalar evxy = x.template segment<2>(3).squaredNorm();
        Scalar evz  = (x(5) - Scalar(vz_ref_)) * (x(5) - Scalar(vz_ref_));
        Scalar bx, by;
        bodyZHorizontal(x, bx, by);
        Scalar eom  = x.template segment<3>(10).squaredNorm();
        return Scalar(0.5) * (
               Scalar(wp_xy_)*ep + Scalar(wp_z_)*ez
             + Scalar(wv_xy_)*evxy + Scalar(wv_z_)*evz
             + Scalar(watt_)*(bx*bx + by*by)
             + Scalar(wom_)*eom);
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.segment(0,2) = Scalar(wp_xy_)*x.segment(0,2);
        g(2)           = Scalar(wp_z_)*x(2);
        g.segment(3,2) = Scalar(wv_xy_)*x.segment(3,2);
        g(5)           = Scalar(wv_z_)*(x(5) - Scalar(vz_ref_));
        Scalar bx, by;
        bodyZHorizontal(x, bx, by);
        Matrix<Scalar> Jtilt = bodyZHorizontalJacobian(x);
        Vector<Scalar> etilt(2);
        etilt << bx, by;
        g.segment(6,4) += Scalar(watt_) * Jtilt.transpose() * etilt;
        g.segment(10,3) = Scalar(wom_)*x.segment(10,3);
        return g;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H(0,0)=H(1,1) = Scalar(wp_xy_);
        H(2,2) = Scalar(wp_z_);
        H(3,3)=H(4,4) = Scalar(wv_xy_);
        H(5,5) = Scalar(wv_z_);
        Matrix<Scalar> Jtilt = bodyZHorizontalJacobian(x);
        H.block(6,6,4,4) += Scalar(watt_) * Jtilt.transpose() * Jtilt;
        H(10,10)=H(11,11)=H(12,12) = Scalar(wom_);
        return H;
    }
};

// ── Stage constraints ─────────────────────────────────────────────────────────
template<typename Scalar>
class FminCon : public StageConstraintBase<Scalar> { public:
    FminCon() { this->constraint_type=ConstraintType::NO; this->dim_c=1; }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return (Vector<Scalar>(1) << Scalar(FMIN)-u(0)).finished(); }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(1, x.size()); }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(1,u.size()); J(0,0)=Scalar(-1); return J; }};

template<typename Scalar>
class FmaxCon : public StageConstraintBase<Scalar> { public:
    FmaxCon() { this->constraint_type=ConstraintType::NO; this->dim_c=1; }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return (Vector<Scalar>(1) << u(0)-Scalar(FMAX)).finished(); }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(1, x.size()); }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(1,u.size()); J(0,0)=Scalar(1); return J; }};

template<typename Scalar>
class MomentCon : public StageConstraintBase<Scalar> { Scalar tau_; public:
    MomentCon() : tau_(static_cast<Scalar>(TAU_MAX)) {
        this->constraint_type=ConstraintType::SOC; this->dim_c=4; }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Vector<Scalar> cn(4); cn << tau_, u(1), u(2), u(3); return -cn; }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(4, x.size()); }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(4,u.size());
        J(1,1)=J(2,2)=J(3,3)=Scalar(-1); return J; }};

template<typename Scalar>
class ThetaBounds : public StageConstraintBase<Scalar> { Scalar lo_, hi_; public:
    ThetaBounds(double lo, double hi) : lo_(lo), hi_(hi) {
        this->constraint_type=ConstraintType::NO; this->dim_c=2; }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Vector<Scalar> cn(2); cn(0)=u(IDX_THETA)-hi_; cn(1)=lo_-u(IDX_THETA); return cn; }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(2, x.size()); }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(2,u.size());
        J(0,IDX_THETA)=Scalar(1); J(1,IDX_THETA)=Scalar(-1); return J; }};

template<typename Scalar>
class ZFloorCon : public StageConstraintBase<Scalar> { public:
    ZFloorCon() { this->constraint_type=ConstraintType::NO; this->dim_c=1; }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return (Vector<Scalar>(1) << -x(2)).finished(); }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(1,x.size()); J(0,2)=Scalar(-1); return J; }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(1, u.size()); }};

template<typename Scalar>
class GeneralSpeedCon : public StageConstraintBase<Scalar> { public:
    GeneralSpeedCon() { this->constraint_type=ConstraintType::SOC; this->dim_c=4; }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Vector<Scalar> cn(4); cn << Scalar(SPD_PHASE0_MAX), x(3), x(4), x(5); return -cn; }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J=Matrix<Scalar>::Zero(4,NX_SS);
        J(1,3)=Scalar(-1); J(2,4)=Scalar(-1); J(3,5)=Scalar(-1); return J; }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(4, NU_SS); }};

// No-aug CT-cSTC interval constraint:
//   integral(H dt) over [t_k, t_k + theta_k] <= eps
// computed by the SAME RK4 quadrature the augmented accumulator used, but
// directly from (x_k, u_k) — no augmented state involved.
template<typename Scalar>
class CtcsIntervalStepIneqCon : public StageConstraintBase<Scalar> {
public:
    CtcsIntervalStepIneqCon(
        std::shared_ptr<Quad6DOFVarTimeRelativePred<Scalar>> dyn,
        double eps)
        : dyn_(std::move(dyn)), eps_(Scalar(eps)) {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> out(1);
        out(0) = Scalar(dyn_->intervalIntegral(x, u)) - eps_;
        return out;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, x.size());
        const double eps = 1e-6;
        for (int i = 0; i < x.size(); ++i) {
            Vector<Scalar> xp = x;
            Vector<Scalar> xm = x;
            xp(i) += Scalar(eps);
            xm(i) -= Scalar(eps);
            J(0, i) = Scalar(dyn_->intervalIntegral(xp, u) -
                             dyn_->intervalIntegral(xm, u)) / Scalar(2.0 * eps);
        }
        return J;
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        const double eps = 1e-6;
        for (int i = 0; i < u.size(); ++i) {
            Vector<Scalar> up = u;
            Vector<Scalar> um = u;
            up(i) += Scalar(eps);
            um(i) -= Scalar(eps);
            J(0, i) = Scalar(dyn_->intervalIntegral(x, up) -
                             dyn_->intervalIntegral(x, um)) / Scalar(2.0 * eps);
        }
        return J;
    }

private:
    std::shared_ptr<Quad6DOFVarTimeRelativePred<Scalar>> dyn_;
    Scalar eps_;
};

inline Eigen::VectorXd ensureNoAugStateSize(const Eigen::VectorXd& x) {
    if (x.size() == NX_SS) return x;
    Eigen::VectorXd out = Eigen::VectorXd::Zero(NX_SS);
    const int n = std::min<int>(x.size(), NX);
    out.head(n) = x.head(n);
    return out;
}

// ── Solver params ─────────────────────────────────────────────────────────────
inline Param getSolverParams() {
    Param p;
    p.reg1_min = 1e-3;
    p.reg2_min = envOrQ("SZMUK_LEGACY_REG2_MIN", 0.1);
    p.mu_mul   = 0.1;
    p.rho      = envOrQ("SZMUK_LEGACY_RHO", 5.0);
    p.rhoT     = envOrQ("SZMUK_LEGACY_RHOT", 5.0);
    p.rho_mul  = envOrQ("SZMUK_LEGACY_RHO_MUL", 10.0);
    p.tolerance = envOrQ("SZMUK_LEGACY_TOL", 1e-6);
    p.max_iter  = envOrInt("SZMUK_LEGACY_MAX_ITER", 180);
    p.is_quaternion_in_state = false;
    p.use_ddp_terms = false;
    return p;
}

// ── Warm-start / validity helpers ─────────────────────────────────────────────
inline void clampWarmControl(Eigen::VectorXd& u) {
    if (u.size() < NU_SS) return;
    u(0) = std::clamp((double)u(0), (double)FMIN, (double)FMAX);
    u(1) = std::clamp((double)u(1), -TAU_MAX, TAU_MAX);
    u(2) = std::clamp((double)u(2), -TAU_MAX, TAU_MAX);
    u(3) = std::clamp((double)u(3), -TAU_MAX, TAU_MAX);
    u(IDX_THETA) = std::clamp((double)u(IDX_THETA), THL, THH);
}

inline bool finiteTrajectory(const std::vector<Eigen::VectorXd>& X,
                             const std::vector<Eigen::VectorXd>& U) {
    for (const auto& x : X) if (!x.allFinite()) return false;
    for (const auto& u : U) if (!u.allFinite()) return false;
    return true;
}

inline bool plausibleTrajectory(const std::vector<Eigen::VectorXd>& X,
                                int max_node, int n_solve) {
    const double dt_limit = static_cast<double>(n_solve) * THH + 1.0;
    const int last = std::min(max_node, static_cast<int>(X.size()) - 1);
    for (int k = 0; k <= last; ++k) {
        const auto& x = X[k];
        if (x.size() <= IDX_DT) return false;
        if (x(2) < -0.05) return false;
        if (x.head(3).norm() > 100.0) return false;
        if (x.segment(3, 3).norm() > 50.0) return false;
        if (x.segment(10, 3).norm() > 50.0) return false;
        if (x(IDX_DT) < -1e-8 || x(IDX_DT) > dt_limit) return false;
    }
    return true;
}

// ── Factory ───────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& x0_rel,
    const std::vector<Eigen::VectorXd>& prev_U = {},
    const std::vector<Eigen::VectorXd>& prev_X = {},
    const std::vector<Eigen::MatrixXd>& prev_K = {},
    std::shared_ptr<ConstantAccelPredictor> predictor = nullptr)
{
    const int N = HORIZON;
    auto problem = std::make_shared<OptimalControlProblem<double>>(N);

    if (!predictor) {
        predictor = std::make_shared<ConstantAccelPredictor>();
    }

    auto dyn = std::make_shared<Quad6DOFVarTimeRelativePred<double>>(predictor);
    dyn->setMass(MASS);
    dyn->setGravity(GRAVITY);
    dyn->setJb(J_B);
    dyn->setTargetAccel(predictor->predictAccel(0.0));

    auto tcost  = std::make_shared<RelTermCost<double>>(
        TERM_W_POS_XY, TERM_W_POS_Z,
        TERM_W_VEL_XY, TERM_W_VEL_Z,
        TERM_W_TILT, TERM_W_OM, TERM_VZ_REF);
    auto cfmin  = std::make_shared<FminCon<double>>();
    auto cfmax  = std::make_shared<FmaxCon<double>>();
    auto cmom   = std::make_shared<MomentCon<double>>();
    auto cth    = std::make_shared<ThetaBounds<double>>(THL, THH);
    auto czfl   = std::make_shared<ZFloorCon<double>>();
    auto cspeed = std::make_shared<GeneralSpeedCon<double>>();
    // The constraint sees the SCALED integral, so the physical budget
    // CTCS_STEP_EPS is divided by CTCS_Y_SCALE (same zero set as unscaled).
    // Hard-wired: this OCP has no other enforcement mode.
    auto cstep = std::make_shared<CtcsIntervalStepIneqCon<double>>(
        dyn, CTCS_STEP_EPS / CTCS_Y_SCALE);

    const double eps_cost = envOrQ("SZMUK_LEGACY_W_EPS", 1.0);
    const Eigen::Vector3d r_follow(0.0, 0.0, Z_STAGE);
    auto stage_cost = std::make_shared<CTcSTCStageCost<double>>(
        eps_cost,
        envOrQ("SZMUK_LEGACY_W_POS_XY", W_STAGE_TARGET_XY),
        envOrQ("SZMUK_LEGACY_W_POS_Z", W_STAGE_TARGET_Z),
        envOrQ("SZMUK_LEGACY_W_VEL", W_STAGE_TARGET_VEL),
        r_follow);

    for (int k = 0; k < N; ++k) {
        problem->setStageDynamics(k, dyn);
        problem->setStageCost(k, stage_cost);
        problem->addStageConstraint(k, cfmin);
        problem->addStageConstraint(k, cfmax);
        problem->addStageConstraint(k, cmom);
        problem->addStageConstraint(k, cth);
        problem->addStageConstraint(k, czfl);
        problem->addStageConstraint(k, cspeed);
        problem->addStageConstraint(k, cstep);
    }
    problem->setTerminalCost(tcost);
    // No CT-cSTC terminal constraint: the interval inequality above is the
    // entire enforcement (the terminal-eq y_N=0 belongs to the augmented arm).

    Eigen::VectorXd x0ss(NX_SS);
    x0ss.setZero();
    x0ss.segment(0, NX) = x0_rel.head(NX);
    x0ss(IDX_DT) = 0.0;
    problem->setInitialState(0, x0ss);

    // Rebase the shifted warm-start references: IDX_DT is a per-solve
    // accumulator, but the core's makeXshifted() does a plain index shift, so
    // prev_X[0] here is the previously executed node carrying the old
    // horizon's t0. Subtract it (benchmark makeXshifted equivalent) and pin
    // the first reference to the new initial state.
    std::vector<Eigen::VectorXd> Xw = prev_X;
    if (!Xw.empty()) {
        const Eigen::VectorXd x_base = ensureNoAugStateSize(Xw[0]);
        const double t0_shift = x_base(IDX_DT);
        bool finite = true;
        for (auto& x : Xw) {
            x = ensureNoAugStateSize(x);
            if (!x.allFinite()) { finite = false; break; }
            x(IDX_DT) = std::max(0.0, x(IDX_DT) - t0_shift);
        }
        if (finite) {
            Xw[0] = x0ss;
        } else {
            Xw.clear();
        }
    }

    const bool use_feedback =
        useRhFeedbackWarmStart() &&
        static_cast<int>(prev_U.size()) >= N &&
        static_cast<int>(Xw.size()) > N &&
        static_cast<int>(prev_K.size()) >= N;

    Eigen::VectorXd sim14 = x0ss;
    for (int k = 0; k < N; ++k) {
        Eigen::VectorXd uk(NU_SS);
        uk.setZero();
        if (static_cast<int>(prev_U.size()) >= N) {
            uk = prev_U[k];
            if (use_feedback) {
                const Eigen::VectorXd& x_ref = Xw[k];
                const Eigen::MatrixXd& K = prev_K[k];
                Eigen::VectorXd dx = sim14 - x_ref;
                Eigen::VectorXd u_fb = uk;
                bool feedback_dim_ok = (K.rows() == NU_SS && K.cols() <= dx.size());
                if (feedback_dim_ok) {
                    u_fb.noalias() = uk + K * dx.head(K.cols());
                }
                Eigen::VectorXd du = Eigen::VectorXd::Zero(NU_SS);
                if (feedback_dim_ok) du = u_fb - uk;
                const double max_du = envOrQ("SZMUK_LEGACY_RH_MAX_FB_DU", 0.05);
                const bool wild_feedback =
                    !feedback_dim_ok ||
                    !u_fb.allFinite() ||
                    !du.allFinite() ||
                    (max_du > 0.0 && du.norm() > max_du) ||
                    u_fb(0) < 0.5 * FMIN ||
                    u_fb(0) > 1.5 * FMAX ||
                    std::abs(u_fb(1)) > 2.0 * TAU_MAX ||
                    std::abs(u_fb(2)) > 2.0 * TAU_MAX ||
                    std::abs(u_fb(3)) > 2.0 * TAU_MAX ||
                    u_fb(IDX_THETA) < 0.0 ||
                    u_fb(IDX_THETA) > 2.0 * THH;
                if (!wild_feedback) {
                    uk = u_fb;
                }
            }
        } else {
            // Geometric straight-line seed. Unlike the augmented arm (soft AL
            // terminal EQ), the interval inequality here is a HARD per-node
            // constraint, so a seed that dives below the staging floor while
            // still outside the staging radius starts the IPM deeply
            // infeasible and stalls the cold solve. Aim at (0,0,Z_STAGE)
            // until laterally captured, and only then at the origin.
            const Eigen::Vector3d p0 = sim14.head(3);
            const Eigen::Vector3d v0 = sim14.segment(3, 3);
            double T_g = static_cast<double>(N - k) * TH0;
            if (T_g < TH0) T_g = TH0;
            const double rxy_seed = p0.head(2).norm();
            Eigen::Vector3d p_goal(0.0, 0.0,
                                   rxy_seed > STAGE_RADIUS ? Z_STAGE : 0.0);
            Eigen::Vector3d a_req = 2.0 * ((p_goal - p0) - v0 * T_g) / (T_g * T_g);
            a_req.z() = std::max(a_req.z(), (-VZ_LAND_MAX - v0.z()) / T_g);
            const double tabs_ws = sim14(IDX_DT);
            const Eigen::Vector3d a_tgt_ws = predictor->predictAccel(tabs_ws);
            const Eigen::Vector3d fw = MASS * (a_req - a_tgt_ws - GRAVITY);
            uk(0) = fw.norm();
            uk(IDX_THETA) = TH0;
        }
        clampWarmControl(uk);

        problem->setInitialControl(k, uk);
        Eigen::VectorXd sim14_next = dyn->propagate14(sim14, uk.segment(0, NU), uk(IDX_THETA));
        sim14_next(2) = std::max(sim14_next(2), 0.0);
        problem->setInitialState(k + 1, sim14_next);
        sim14 = sim14_next;
    }

    return problem;
}

// ── Descriptor ────────────────────────────────────────────────────────────────
inline OCPDescriptor descriptor() {
    OCPDescriptor d;
    d.name = "stc_landing_noaug";
    d.dt = TH0;
    d.default_n_replay = NEX;
    d.default_mass_kg = MASS;
    d.warm_start = OCPDescriptor::WarmStart::Feedback;
    d.command_mode = OCPDescriptor::CommandMode::CmdFullState;
    d.drone_odom_mode = OCPDescriptor::DroneOdomMode::Absolute;
    d.variable_dt = true;
    d.state_dim = 13;
    d.control_dim = 4;
    d.skip_altitude_validation = true;
    d.skip_trajectory_validation = true;
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
        // Ported benchmark validity gates: refuse nonfinite/implausible plans
        // so PlannerCore rejects the solve and restores the warm-start snapshot.
        if (r.success &&
            (!finiteTrajectory(r.state_trajectory, r.control_trajectory) ||
             !plausibleTrajectory(r.state_trajectory,
                                  std::min(NEX, HORIZON), HORIZON))) {
            r.success = false;
        }
    };
    d.prepare_extra = [](const PlannerConfig&, double, const TargetSnapshot& tgt_snapshot) {
        StcLandingNoAugLegacyExtra ex;
        ex.predictor = std::make_shared<ConstantAccelPredictor>();
        ex.predictor->setState(tgt_snapshot.position,
                               tgt_snapshot.velocity,
                               tgt_snapshot.acceleration);
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
    d.sanitize_warm_control = [](Eigen::VectorXd& u) { clampWarmControl(u); };
    d.getSolverParams = getSolverParams;
    d.create = [](const OCPCreateArgs& a) {
        StcLandingNoAugLegacyExtra ex;
        if (a.extra.has_value()) {
            try { ex = std::any_cast<StcLandingNoAugLegacyExtra>(a.extra); }
            catch (const std::bad_any_cast&) {}
        }
        auto pred = ex.predictor;
        if (!pred) {
            pred = std::make_shared<ConstantAccelPredictor>();
            pred->setState(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                           a.target_accel);
        }
        return create(a.current_state, a.prev_U, a.prev_X, a.prev_K, pred);
    };
    return d;
}

}  // namespace StcLandingNoAugLegacyOCP
