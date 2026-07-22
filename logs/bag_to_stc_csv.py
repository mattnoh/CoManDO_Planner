#!/usr/bin/env python3
"""Convert a CoManDO MAVROS flight rosbag into the CSV pair that the paper
plotting scripts (plot_stc_paper.py / animate_stc_paper.py) consume.

Reads:
  /mavros/local_position/odom   drone pose (world ENU) + twist (body frame)
  /target/odom                  target pose/velocity (world ENU)

Writes into --out (default: <bag_dir>/<bag_stem>_analysis/):
  moving_executed.csv   executed flight resampled on drone-odom stamps,
                        with target states interpolated onto the same stamps
                        and the CT-cSTC trigger signals recomputed from the
                        same formulas/constants as ocp_stc_landing.hpp
  constraints.csv       bound values exported for load_current_bounds(), so
                        the figures show THIS branch's constraints (including
                        any SZMUK_* env overrides active when you run this —
                        run it with the same exports you flew with)

Usage:
  python3 bag_to_stc_csv.py <flight.bag> [--t0 SEC] [--t1 SEC] [--out DIR]

  --t0/--t1 are seconds relative to the start of the drone-odom stream,
  to trim pre-trigger hover or post-touchdown sitting from the figures.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import rosbag


# ── Constants mirrored from include/ocp/ocp_stc_landing.hpp ──────────────────
# (env_name, default) — resolved the same way as envOrQ() so figures match a
# run that used SZMUK_* tunables, provided you export them here too.
# NOTE: z_stage and los_alt_trig defaults mirror the planner_launch.launch
# <env> overrides (currently 1.4 / 1.2), NOT the code's 1.8; the remaining
# defaults mirror ocp_stc_landing_noaug.hpp's compiled-in benchmark values
# (faithful-port restore, 2026-07-23). Keep the three places in sync, and
# export the same SZMUK_* env when re-analyzing a flight that overrode them.
_ENV_CONSTANTS = {
    "alt_trig":            ("SZMUK_ALT_TRIG", 0.8),
    "cap_radius":          ("SZMUK_RH_CAP_LAND_RADIUS", 0.45),
    "z_stage":             ("SZMUK_Z_STAGE", 1.4),
    "stage_radius":        ("SZMUK_STAGE_RADIUS", 1.10),
    "stage_trigger_scale": ("SZMUK_STAGE_TRIGGER_SCALE", 0.25),
    "stage_trigger_floor": ("SZMUK_STAGE_TRIGGER_FLOOR", 0.0),
    "los_alt_trig":        ("SZMUK_LOS_ALT_TRIG", 1.2),
    "los_trigger_scale":   ("SZMUK_LOS_TRIGGER_SCALE", 0.25),
    "los_trigger_floor":   ("SZMUK_LOS_TRIGGER_FLOOR", 0.0),
    "los_cone_deg":        ("SZMUK_LOS_CONE_DEG", 30.0),
    "v_phase0_max":        ("SZMUK_SPD_PHASE0_MAX", 3.0),
    "v_land_stc_max":      ("SZMUK_SPD_STC_MAX", 0.75),
    "omega_phase0_max":    ("SZMUK_OMEGA_PHASE0_MAX", 0.35),
    "omega_land_stc_max":  ("SZMUK_OMEGA_STC_MAX", 0.019),
    "tilt_phase0_deg":     ("SZMUK_TILT_PHASE0_DEG", 11.0),
    "gs_cone_deg":         ("SZMUK_GS_CONE_DEG", 15.0),
}
_FIXED_CONSTANTS = {
    "gs_apex_offset": 0.1,     # GS_APEX_OFFSET
    "fmin": 0.08,              # FMIN
    "fmax": 0.60,              # FMAX
    "fmin_land": 0.21,         # T_MIN_AFT
    "fmax_land": 0.40,         # T_MAX_AFT
}


def _env_or(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return float(default)
    try:
        return float(val)
    except ValueError:
        return float(default)


def resolve_constants() -> dict:
    c = {k: _env_or(env, d) for k, (env, d) in _ENV_CONSTANTS.items()}
    c.update(_FIXED_CONSTANTS)
    # tilt_land_stc_deg is the *effective* tightened tilt bound:
    # sin(0.5 * ALPHA_THETA_STC * THETA_STC_DEG) in the integrand.
    c["tilt_land_stc_deg"] = (_env_or("SZMUK_ALPHA_THETA_STC", 1.0)
                              * _env_or("SZMUK_TILT_STC_DEG", 2.0))
    c["gs_cone_tan"] = math.tan(math.radians(c["gs_cone_deg"]))
    c["los_cone_tan"] = math.tan(math.radians(c["los_cone_deg"]))
    return c


def quat_rotmat(qw, qx, qy, qz):
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) or 1.0
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)],
    ])


def read_bag(bag_path: Path):
    drone, target = [], []
    # mavros header stamps are garbage when its time sync is broken (e.g. it
    # was started under a stale /use_sim_time). Trust a header stamp only if
    # it is within 10 s of the bag record time; otherwise fall back to the
    # bag record time, which is always wall clock.
    def stamp_of(m, bag_t):
        h = m.header.stamp.to_sec()
        return h if abs(h - bag_t.to_sec()) < 10.0 else bag_t.to_sec()

    with rosbag.Bag(str(bag_path)) as bag:
        for _, m, bt in bag.read_messages(topics=["/mavros/local_position/odom"]):
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            v = m.twist.twist.linear
            w = m.twist.twist.angular
            # mavros local_position/odom twist is in the child frame
            # (base_link); rotate linear velocity into the world frame.
            vw = quat_rotmat(q.w, q.x, q.y, q.z) @ np.array([v.x, v.y, v.z])
            drone.append((stamp_of(m, bt),
                          p.x, p.y, p.z, vw[0], vw[1], vw[2],
                          q.w, q.x, q.y, q.z, w.x, w.y, w.z))
        for _, m, bt in bag.read_messages(topics=["/target/odom"]):
            p = m.pose.pose.position
            v = m.twist.twist.linear
            target.append((stamp_of(m, bt),
                           p.x, p.y, p.z, v.x, v.y, v.z))
    if not drone:
        raise SystemExit(f"{bag_path}: no /mavros/local_position/odom messages")
    if not target:
        raise SystemExit(f"{bag_path}: no /target/odom messages "
                         "(was target_launch running during the recording?)")
    return np.array(drone), np.array(target)


def compute_triggers(rxyz: np.ndarray, c: dict):
    rxy = np.linalg.norm(rxyz[:, :2], axis=1)
    rz = rxyz[:, 2]

    # landingAltitudeTrigger(): altitude ramp gated by the lateral capture cap.
    trig_alt = np.maximum(0.0, c["alt_trig"] - rz)
    cap2 = c["cap_radius"] ** 2
    trig_cap = np.maximum(0.0, cap2 - rxy**2) / cap2
    a_land = (trig_alt / c["alt_trig"]) * trig_cap

    # Stage trigger, default "radius" mode: active while laterally outside
    # STAGE_RADIUS of the platform.
    margin = (rxy - c["stage_radius"]) / c["stage_trigger_scale"]
    a_stage = np.where(margin > 0.0, c["stage_trigger_floor"] + margin, 0.0)

    # LOS trigger: active below LOS_ALT_TRIG relative altitude.
    los_margin = (c["los_alt_trig"] - rz) / c["los_trigger_scale"]
    a_los = np.where(los_margin > 0.0, c["los_trigger_floor"] + los_margin, 0.0)

    return a_land, a_stage, a_los


def convert(bag_path: Path, out_dir: Path, t0: float | None = None,
            t1: float | None = None) -> Path:
    """Convert bag → out_dir/{moving_executed.csv, constraints.csv}.

    t0/t1 trim in seconds relative to the first drone-odom message.
    Returns the path of the written moving_executed.csv.
    """
    drone, target = read_bag(bag_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample on drone-odom stamps, restricted to the overlap with target data
    # and any t0/t1 trim.
    t = drone[:, 0]
    t_rel = t - t[0]
    lo = max(t[0], target[0, 0])
    hi = min(t[-1], target[-1, 0])
    keep = (t >= lo) & (t <= hi)
    if t0 is not None:
        keep &= t_rel >= t0
    if t1 is not None:
        keep &= t_rel <= t1
    drone = drone[keep]
    t = drone[:, 0]
    if len(t) < 2:
        raise SystemExit("trim/overlap left fewer than 2 samples")

    tgt = np.column_stack([
        np.interp(t, target[:, 0], target[:, k]) for k in range(1, 7)
    ])

    xyz = drone[:, 1:4]
    vxyz = drone[:, 4:7]
    quat = drone[:, 7:11]
    omega = drone[:, 11:14]
    rxyz = xyz - tgt[:, 0:3]
    rvxyz = vxyz - tgt[:, 3:6]

    # Stop at touchdown: cut everything after relative altitude first reaches
    # zero — beyond that the drone is down/disarmed and the target driving
    # away just pollutes the figures.
    touch = np.flatnonzero(rxyz[:, 2] <= 0.01)
    if len(touch) and touch[0] >= 1:
        end = int(touch[0]) + 1
        t, xyz, vxyz, quat, omega = t[:end], xyz[:end], vxyz[:end], quat[:end], omega[:end]
        tgt, rxyz, rvxyz = tgt[:end], rxyz[:end], rvxyz[:end]

    c = resolve_constants()
    a_land, a_stage, a_los = compute_triggers(rxyz, c)

    header = ("t_acc,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,fz_B,"
              "tx,ty,tz,tvx,tvy,tvz,rx,ry,rz,rvx,rvy,rvz,"
              "ctcs_a_altitude_state,ctcs_a_stage,ctcs_a_los")
    rows = np.column_stack([
        t - t[0], xyz, vxyz, quat, omega,
        np.full((len(t), 1), np.nan),          # fz_B: not observable from the bag
        tgt[:, 0:3], tgt[:, 3:6], rxyz, rvxyz,
        a_land[:, None], a_stage[:, None], a_los[:, None],
    ])
    csv_path = out_dir / "moving_executed.csv"
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="",
               fmt="%.6f")

    con_path = out_dir / "constraints.csv"
    with open(con_path, "w") as f:
        f.write("param,value\n")
        for k, v in sorted(c.items()):
            f.write(f"{k},{v}\n")

    dur = t[-1] - t[0]
    print(f"wrote {csv_path}  ({len(t)} samples, {dur:.1f} s)")
    print(f"wrote {con_path}")
    print(f"triggers first active [s]: "
          f"a_stage_off={_last_on(t, a_stage)}  "
          f"a_los_on={_first_on(t, a_los)}  a_land_on={_first_on(t, a_land)}")
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("--t0", type=float, default=None,
                    help="trim start, seconds from first drone-odom message")
    ap.add_argument("--t1", type=float, default=None,
                    help="trim end, seconds from first drone-odom message")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default <bag_dir>/<bag_stem>_analysis)")
    args = ap.parse_args()
    out_dir = args.out or args.bag.parent / f"{args.bag.stem}_analysis"
    convert(args.bag, out_dir, args.t0, args.t1)


def _first_on(t, a, eps=1e-9):
    idx = np.flatnonzero(a > eps)
    return round(float(t[idx[0]] - t[0]), 2) if len(idx) else None


def _last_on(t, a, eps=1e-9):
    idx = np.flatnonzero(a > eps)
    return round(float(t[idx[-1]] - t[0]), 2) if len(idx) else None


if __name__ == "__main__":
    main()
