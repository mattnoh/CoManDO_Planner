/// @file tools_gs_probe.cpp
/// @brief Offline STC screening probe for stc_landing_noaug (not a unit test).
///
/// Cold-solves the SITL scenario (R=1 circle, w=0.4, pad z=0.2) from 8 target
/// phases at a given world hover altitude and reports, per config:
///   ok/8 convergent activations, worst PLANNED glideslope angle below
///   ALT_TRIG, worst planned tilt, and solve times.
/// [OK-RT] marks configs that keep ms_max <= 250 (the RH replan budget) with
/// >=7/8 convergence — ALWAYS screen a candidate here before flying it; the
/// single-horizon tuning recipe blows the RH real-time budget if copied
/// verbatim (see logs/<run>/tuning.env of the 2026-07-23 horizon-tuned run).
///
/// Not built by CMake. Build standalone:
///   g++ -std=c++17 -O3 -DNDEBUG -Iinclude -I../ALIPDDP-main/alipddp \
///       -I../ALIPDDP-main/problem_maker -I/usr/include/eigen3 \
///       test/tools_gs_probe.cpp src/quadrotor_mpc.cpp \
///       ../ALIPDDP-main/alipddp/alipddp/alipddp.cpp \
///       ../ALIPDDP-main/problem_maker/optimal_control_problem.cpp -o /tmp/gs_probe
///   SZMUK_Z_STAGE=1.55 SZMUK_LOS_ALT_TRIG=1.2 SZMUK_RH_N=40 \
///   SZMUK_MAX_ITER=120 SZMUK_THH=0.16 SZMUK_W_LAND_GS=20000 /tmp/gs_probe 1.80

// Cold solves around the circle. For converged plans report:
//  - lat at the moment the plan drops through the staging floor (how vertical)
//  - worst PLANNED glideslope angle below ALT_TRIG (cone compliance)
//  - worst planned tilt anywhere (phase-0 cost of the tighter capture)
#include "planner_core/quadrotor_mpc.hpp"
#include "planner_core/ocp_registry.hpp"
#include "ocp/ocp_stc_landing_noaug.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){
    const double R=1.0,W=0.4,CZ=0.2;
    const double zw=(argc>1)?atof(argv[1]):1.80;
    const double ZS=StcLandingNoAugOCP::Z_STAGE, AT=StcLandingNoAugOCP::ALT_TRIG;
    int ok=0, physical_ok=0;
    double lat_sum=0, gs_worst=0, tilt_worst=0, lat_worst=0, ms_sum=0, ms_max=0;
    double speed_worst=0, land_speed_worst=0, omega_worst=0, land_omega_worst=0;
    double thrust_min=1e9, thrust_max=-1e9, moment_worst=0, z_min=1e9, terminal_worst=0;
    for(int k=0;k<8;++k){
        const double ph=k*M_PI/4.0;
        const Eigen::Vector3d tp(R*cos(ph),R*sin(ph),CZ);
        const Eigen::Vector3d tv(-R*W*sin(ph),R*W*cos(ph),0.0);
        const Eigen::Vector3d ta(-R*W*W*cos(ph),-R*W*W*sin(ph),0.0);
        Eigen::VectorXd x0=Eigen::VectorXd::Zero(13);
        x0.segment(0,3)=Eigen::Vector3d(2.0,2.0,zw)-tp; x0.segment(3,3)=-tv; x0(6)=1.0;
        auto pred=std::make_shared<StcLandingNoAugOCP::ConstantAccelPredictor>();
        pred->setState(tp,tv,ta);
        StcLandingNoAugOCP::StcLandingNoAugExtra ex; ex.predictor=pred;
        QuadrotorMPC::Config cfg; cfg.ocp_type="stc_landing_noaug"; cfg.n_shift=7;
        QuadrotorMPC mpc(cfg);
        auto r=mpc.solve(x0,ta,std::any(ex),0.0);
        if(!(r.success&&r.constraint_error<1.0)) continue;
        ok++; ms_sum+=r.solve_time_ms; ms_max=std::max(ms_max,r.solve_time_ms);
        double lat_at_floor=-1,gsm=0,tlt=0,spd=0,lspd=0,om=0,lom=0,zlo=1e9;
        for(const auto&x:r.state_trajectory){
            const double lat=std::hypot(x(0),x(1));
            if(lat_at_floor<0 && x(2)<ZS) lat_at_floor=lat;
            if(x(2)<AT && x(2)>0.15)
                gsm=std::max(gsm, std::atan2(lat,x(2)+0.1)*180.0/M_PI);
            tlt=std::max(tlt, 2.0*std::asin(std::min(1.0,std::hypot(x(7),x(8))))*180.0/M_PI);
            const double v=x.segment(3,3).norm(), w=x.segment(10,3).norm();
            spd=std::max(spd,v); om=std::max(om,w); zlo=std::min(zlo,x(2));
            if(x(2)<AT){ lspd=std::max(lspd,v); lom=std::max(lom,w); }
        }
        double umin=1e9,umax=-1e9,mom=0;
        for(const auto&u:r.control_trajectory){
            umin=std::min(umin,u(0)); umax=std::max(umax,u(0));
            mom=std::max(mom,u.segment(1,3).norm());
        }
        if(lat_at_floor<0) lat_at_floor=0;
        lat_sum+=lat_at_floor; lat_worst=std::max(lat_worst,lat_at_floor);
        gs_worst=std::max(gs_worst,gsm); tilt_worst=std::max(tilt_worst,tlt);
        speed_worst=std::max(speed_worst,spd); land_speed_worst=std::max(land_speed_worst,lspd);
        omega_worst=std::max(omega_worst,om); land_omega_worst=std::max(land_omega_worst,lom);
        thrust_min=std::min(thrust_min,umin); thrust_max=std::max(thrust_max,umax);
        moment_worst=std::max(moment_worst,mom); z_min=std::min(z_min,zlo);
        terminal_worst=std::max(terminal_worst,r.state_trajectory.back().head(6).norm());
        const bool hard_physical =
            spd<=StcLandingNoAugOCP::SPD_PHASE0_MAX+1e-3 &&
            umin>=StcLandingNoAugOCP::FMIN-1e-3 && umax<=StcLandingNoAugOCP::FMAX+1e-3 &&
            mom<=StcLandingNoAugOCP::TAU_MAX+1e-3;
        if(hard_physical) physical_ok++;
    }
    printf("RESULT ok=%d/8 phys=%d/%d gs=%.1fdeg tilt=%.1fdeg "
           "speed=%.2f land_speed=%.2f omega=%.3f land_omega=%.3f "
           "thrust=[%.3f,%.3f] moment=%.1f z_min=%.3f term6=%.3f "
           "ms_avg=%.0f ms_max=%.0f %s\n",
           ok, physical_ok, ok, gs_worst, tilt_worst,
           speed_worst, land_speed_worst, omega_worst, land_omega_worst,
           thrust_min, thrust_max, moment_worst, z_min, terminal_worst,
           ok?ms_sum/ok:0.0, ms_max,
           (ms_max<=250.0&&physical_ok==ok&&ok>=7)?"[OK-PHYS-RT]":"");
    return 0;
}
