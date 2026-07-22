#!/usr/bin/env python3
"""Planned-state panels for a receding-horizon CT-cSTC landing run.

Companion to plot_stc_paper.py, which plots the EXECUTED flight from the bag.
This plots what the SOLVER planned, from all_solves.csv, so the two can be
compared without mistaking tracking error for constraint violation.

Styling, colours, panel geometry, bound drawing and the LOS-angle maths are
IMPORTED from plot_stc_paper so both figures are visually identical and stay
in sync if that file is restyled. Only what is genuinely different lives here:
reading per-solve horizons instead of one executed time series.

Two traces per panel:
  * bold  — the committed plan: the executed prefix (first NEX nodes) of each
            solve stitched on a global clock. This is what was handed to the
            vehicle.
  * faint — every full horizon from its own solve time, showing what the
            planner intended beyond the commit point.

The armed region of each trigger is shaded. The landing consequences ride on a
COMPOUND trigger (below ALT_TRIG *and* within CAP_LAND_RADIUS laterally), so a
sample above the bound while the trigger is off is not a violation.

Usage:
  python3 plot_planned_states.py <run_dir> [<run_dir> ...]
  python3 plot_planned_states.py --all          # every stc_landing_noaug_* run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Shared look-and-feel + maths from the paper plotter (import is safe: its
# entry point is guarded by __main__).
from plot_stc_paper import (  # noqa: E402
    C_STATE, C_UPPER, C_TRIG_ALT, C_TRIG_THR, C_TRIG_STAGE,
    X_FS, Y_FS, LEG_FS,
    panel_constraint, compute_los_angle_deg, first_trigger_time,
)

C_GHOST = "#a8c8e6"     # unexecuted horizon tails
C_SHADE = "#ffe2b8"     # trigger-armed band

# Defaults of ocp_stc_landing_noaug.hpp; constraints.csv overrides per run.
DEFAULTS = dict(
    alt_trig=0.8, cap_radius=0.45,
    gs_cone_deg=15.0, gs_apex_offset=0.1,
    los_cone_deg=30.0, los_alt_trig=1.3, los_trigger_scale=0.25,
    omega_land_stc_max=0.05, omega_phase0_max=0.5,
    tilt_land_stc_deg=2.0, tilt_phase0_deg=15.0,
    v_land_stc_max=0.75, v_phase0_max=3.0,
    z_stage=1.3, stage_radius=0.55, stage_trigger_scale=0.11,
)


def load_bounds(run: Path) -> dict:
    b = dict(DEFAULTS)
    f = run / "constraints.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            b[str(r["param"])] = float(r["value"])
    b["gs_cone_tan"] = b.get("gs_cone_tan", np.tan(np.radians(b["gs_cone_deg"])))
    return b


# ── Triggers (mirror evalCtcsIntegrandTerms in ocp_stc_landing_noaug.hpp) ─────
def landing_trigger(pz, rxy, b):
    cap = b["cap_radius"]
    ta = np.maximum(0.0, b["alt_trig"] - pz) / b["alt_trig"]
    tc = np.maximum(0.0, cap * cap - rxy * rxy) / (cap * cap)
    return ta * tc


def los_trigger(pz, b):
    m = (b["los_alt_trig"] - pz) / b["los_trigger_scale"]
    return np.where(m > 0.0, m, 0.0)


def stage_trigger(rxy, b):
    m = (rxy - b["stage_radius"]) / b["stage_trigger_scale"]
    return np.where(m > 0.0, m, 0.0)


def derive(g: pd.DataFrame, b: dict) -> pd.DataFrame:
    p = g[["px", "py", "pz"]].to_numpy()
    q = g[["qw", "qx", "qy", "qz"]].to_numpy()
    d = pd.DataFrame(index=g.index)
    d["rxy"] = np.hypot(p[:, 0], p[:, 1])
    d["pz"] = p[:, 2]
    d["speed"] = np.linalg.norm(g[["vx", "vy", "vz"]].to_numpy(), axis=1)
    d["omega"] = np.linalg.norm(g[["wx", "wy", "wz"]].to_numpy(), axis=1)
    d["tilt"] = np.degrees(2 * np.arcsin(np.clip(np.hypot(q[:, 1], q[:, 2]), 0, 1)))
    d["los"] = compute_los_angle_deg(p, q)          # shared with the paper figure
    d["gs"] = np.degrees(np.arctan2(d["rxy"], d["pz"] + b["gs_apex_offset"]))
    d["a_land"] = landing_trigger(d["pz"].to_numpy(), d["rxy"].to_numpy(), b)
    d["a_los"] = los_trigger(d["pz"].to_numpy(), b)
    d["a_stage"] = stage_trigger(d["rxy"].to_numpy(), b)
    return d


def split_horizons(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split all_solves.csv into individual horizons.

    solve_num is NOT sufficient: the logger initialises on the session's first
    profile, so a hover-then-land run writes the 101-node hover plan and the
    first 31-node landing horizon under the SAME solve_num 0 — and the hover
    plan is in the WORLD frame while the landing is target-relative. Segment on
    the node counter restarting, then keep only the modal length (the landing
    horizon), which drops the hover plan.
    """
    node = df["node"].to_numpy()
    starts = np.flatnonzero(node == 0)
    segs = [df.iloc[a:b].reset_index(drop=True)
            for a, b in zip(starts, list(starts[1:]) + [len(df)])]
    if not segs:
        return []
    keep = int(pd.Series([len(s) for s in segs]).mode().iloc[0])
    return [s for s in segs if len(s) == keep]


def build_timeline(run: Path, b: dict, nex: int):
    df = pd.read_csv(run / "all_solves.csv")
    committed, horizons, t0 = [], [], 0.0
    for g in split_horizons(df):
        g = g.sort_values("node").reset_index(drop=True)
        d = derive(g, b)
        d["t"] = g["t"].to_numpy() + t0
        horizons.append(d)
        k = min(nex, len(g) - 1)
        # Nodes 0..k-1 only: node k belongs to the NEXT solve's node 0 (same
        # instant), and including both would draw a spurious vertical step at
        # every commit boundary.
        committed.append(d.iloc[:k])
        t0 += float(g["t"].iloc[k])
    return pd.concat(committed, ignore_index=True), horizons


def add_planned(ax, horizons, plan, key, trig_key, shade=False):
    """Overlay the planned traces onto a paper-style panel.

    shade=True bands the region where the trigger is actually armed. Off by
    default: the trigger lines drawn by panel_constraint already mark the
    switch, matching the executed-flight figure.
    """
    if shade:
        armed = plan[trig_key].to_numpy() > 1e-9
        if armed.any():
            ax.fill_between(plan["t"].to_numpy(), 0, 1, where=armed,
                            transform=ax.get_xaxis_transform(),
                            color=C_SHADE, alpha=0.55, lw=0, zorder=0,
                            label="Constraint armed")
    for h in horizons:
        ax.plot(h["t"].to_numpy(), h[key].to_numpy(),
                c=C_GHOST, lw=0.8, alpha=0.75, zorder=2, label="Full horizons")


def plot_run(run: Path, nex: int, dpi: int, shade: bool = False) -> Path:
    b = load_bounds(run)
    plan, horizons = build_timeline(run, b, nex)
    t = plan["t"].to_numpy()

    land_t = first_trigger_time(t, plan["a_land"].to_numpy())
    los_t = first_trigger_time(t, plan["a_los"].to_numpy())
    stage_t = first_trigger_time(t, plan["a_stage"].to_numpy())
    # draw_trigger_lines expects (label, t, colour) triples.
    trig_land = [("$a_{land}$", land_t, C_TRIG_ALT)] if land_t is not None else []
    trig_los = [("$a_{los}$", los_t, C_TRIG_THR)] if los_t is not None else []
    trig_stage = [("$a_{stage}$", stage_t, C_TRIG_STAGE)] if stage_t is not None else []

    # Same geometry as plot_stc_paper's FIGURE 2 so the pair reads as a set.
    FIGS_W, FIGS_H = 16.0, 8.5
    PW, PH, LEFT, GAP = 0.220, 0.350, 0.080, 0.090
    ROW_Y = (0.480, 0.080)
    fig = plt.figure(figsize=(FIGS_W, FIGS_H), facecolor="white")
    x0s = [LEFT + i * (PW + GAP) for i in range(3)]

    spec = [
        (0, 0, "speed", "Speed, $v=\\|\\dot r\\|_2$ [m/s]", "a_land", land_t,
         b["v_phase0_max"], b["v_land_stc_max"], None, None, trig_land),
        (1, 0, "los", "LOS cone angle, $\\theta_{los}$ [deg]", "a_los", los_t,
         None, b["los_cone_deg"], None, None, trig_los),
        (2, 0, "gs", "Glideslope angle, $\\theta_{gs}$ [deg]", "a_land", land_t,
         None, b["gs_cone_deg"], None, None, trig_land),
        (0, 1, "omega", "Angular rate, $\\|\\omega\\|_2$ [rad/s]", "a_land", land_t,
         b["omega_phase0_max"], b["omega_land_stc_max"], None, None, trig_land),
        (1, 1, "tilt", "Tilt, $\\theta$ [deg]", "a_land", land_t,
         b["tilt_phase0_deg"], b["tilt_land_stc_deg"], None, None, trig_land),
        (2, 1, "pz", "Stage altitude, $r_z$ [m]", "a_stage", stage_t,
         None, None, b["z_stage"], None, trig_stage),
    ]

    axes = []
    for col, row, key, ylabel, trig_key, sw, up_pre, up_post, lo_pre, lo_post, trg in spec:
        ax = fig.add_axes([x0s[col], ROW_Y[row], PW, PH])
        add_planned(ax, horizons, plan, key, trig_key, shade=shade)
        panel_constraint(ax, t, plan[key].to_numpy(), ylabel, sw,
                         upper_pre=up_pre, upper_post=up_post,
                         lower_pre=lo_pre, lower_post=lo_post,
                         triggers=trg, label_state="Planned (committed)")
        if row == 1:
            ax.set_xlabel("Planned time [s]", fontsize=X_FS)
        axes.append(ax)

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=LEG_FS, mode=None,
               bbox_to_anchor=(LEFT + (3 * PW + 2 * GAP) / 2, 0.945),
               frameon=True, framealpha=0.92, edgecolor="0.7",
               handlelength=1.8, columnspacing=1.0, borderpad=0.5)
    fig.align_ylabels([axes[0], axes[3]])
    fig.align_ylabels([axes[1], axes[4]])
    fig.align_ylabels([axes[2], axes[5]])

    out = run / "planned_states.png"
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run directories (default: --all)")
    ap.add_argument("--all", action="store_true",
                    help="every stc_landing_noaug_* run in this directory")
    ap.add_argument("--nex", type=int, default=7, help="SZMUK_RH_NEX (default 7)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--shade-armed", action="store_true",
                    help="band the trigger-armed region (off by default)")
    args = ap.parse_args()

    if args.all or not args.runs:
        runs = sorted(HERE.glob("stc_landing_noaug_*"))
    else:
        runs = [Path(a) if Path(a).is_absolute() else HERE / a for a in args.runs]
    if not runs:
        raise SystemExit("no run directories found")
    for r in runs:
        if not (r / "all_solves.csv").exists():
            print(f"skip {r.name}: no all_solves.csv")
            continue
        print("wrote", plot_run(r, args.nex, args.dpi, args.shade_armed))


if __name__ == "__main__":
    main()
