#!/usr/bin/env python3
"""
trajectory_analysis_plots.py — Receding-horizon quadrotor flight analysis
==========================================================================
Produces a single multi-page vector PDF with 4 figures:
  Fig 1 — 3D trajectory comparison (4 orbital views)
           actual path + first-solve plan + 10 sampled body-frame tripods
  Fig 2 — State tracking: position (x/y/z together), velocity, quaternion, ang-rate
  Fig 3 — Tracking error time series (same grouping)
  Fig 4 — Error distribution box plots

CSV layout (planner_node.cpp convention):
  all_solves.csv        solve_num, solve_time_ms, node,
                        t, x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz
  commanded_state.csv   timestamp, x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz
  actual_state.csv      timestamp, x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz

Usage:
  python3 trajectory_analysis_plots.py <folder>
  python3 trajectory_analysis_plots.py <folder> --cutoff 10 --out report.pdf
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning)

# ── Column names ───────────────────────────────────────────────────────────────
SOLVE_COLS = ["solve_num","solve_time_ms","node",
              "t","x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz"]
CMD_COLS   = ["timestamp","x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz"]
ACT_COLS   = ["timestamp","x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz"]

# ── Colours ────────────────────────────────────────────────────────────────────
FIRST_C = "#D97706"   # first-solve plan
CMD_C   = "#1D4ED8"   # commanded
ACT_C   = "#DC2626"   # actual
ERR_C   = "#7C3AED"   # error
BF_X    = "#EF4444"   # body-frame X
BF_Y    = "#22C55E"   # body-frame Y
BF_Z    = "#3B82F6"   # body-frame Z
GRID_C  = "#E5E7EB"
TEXT_C  = "#111827"
DIM_C   = "#6B7280"
FILL_A  = 0.10

# Per-group, per-variable colours
GRP_COLS = {
    "position":     {"x":"#1D4ED8","y":"#16A34A","z":"#DC2626"},
    "velocity":     {"vx":"#1D4ED8","vy":"#16A34A","vz":"#DC2626"},
    "quaternion":   {"qw":"#7C3AED","qx":"#1D4ED8","qy":"#16A34A","qz":"#DC2626"},
    "angular_rate": {"wx":"#1D4ED8","wy":"#16A34A","wz":"#DC2626"},
}

FS = 11

VIEWS = [
    dict(elev=25, azim=60,   label="Azim +60 deg"),
    dict(elev=25, azim=120,  label="Azim +120 deg"),
    dict(elev=25, azim=-60,  label="Azim -60 deg"),
    dict(elev=25, azim=-120, label="Azim -120 deg"),
]

plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#CBD5E1",
    "axes.grid":           True,
    "grid.color":          GRID_C,
    "grid.linewidth":      0.55,
    "grid.linestyle":      "--",
    "xtick.color":         DIM_C,
    "ytick.color":         DIM_C,
    "xtick.labelsize":     FS - 1,
    "ytick.labelsize":     FS - 1,
    "font.size":           FS,
    "axes.labelsize":      FS,
    "axes.titlesize":      FS + 1,
    "axes.labelcolor":     TEXT_C,
    "axes.titlecolor":     TEXT_C,
    "axes.titleweight":    "semibold",
    "legend.frameon":      True,
    "legend.edgecolor":    "#CBD5E1",
    "legend.facecolor":    "white",
    "legend.fontsize":     FS - 2,
    "lines.linewidth":     1.8,
    "savefig.facecolor":   "white",
    "savefig.transparent": False,
    "pdf.fonttype":        42,   # TrueType embed — fully vector text
    "ps.fonttype":         42,
})


# ===============================================================================
#  Data loading
# ===============================================================================

def load_logs(folder: str, cutoff=None) -> dict:
    folder = Path(folder)

    def _read(fname, cols):
        path = folder / fname
        if not path.exists():
            print(f"  [warn] {fname} not found")
            return None
        try:
            raw = pd.read_csv(path, header=0)
            n   = min(len(cols), raw.shape[1])
            raw = raw.iloc[:, :n].copy()
            raw.columns = cols[:n]
            if cutoff is not None:
                tc = "t" if "t" in raw.columns else "timestamp"
                if tc in raw.columns:
                    t0  = raw[tc].iloc[0]
                    raw = raw[raw[tc] - t0 <= cutoff].reset_index(drop=True)
            return raw
        except Exception as exc:
            print(f"  [warn] {fname}: {exc}")
            return None

    solves = _read("all_solves.csv",      SOLVE_COLS)
    cmd    = _read("commanded_state.csv", CMD_COLS)
    act    = _read("actual_state.csv",    ACT_COLS)

    if solves is not None:
        solves["solve_num"] = solves["solve_num"].astype(int)
        solves["node"]      = solves["node"].astype(int)

    return dict(solves=solves, cmd=cmd, act=act)


def extract_solves(logs: dict) -> dict:
    solves = logs["solves"]
    if solves is None:
        return {}
    solve_map = {}
    for sn, grp in solves.groupby("solve_num", sort=True):
        solve_map[int(sn)] = grp.sort_values("node").reset_index(drop=True)
    solve_nums  = np.array(sorted(solve_map.keys()))
    first_solve = solve_map[int(solve_nums[0])] if len(solve_nums) else None
    return dict(solve_map=solve_map, solve_nums=solve_nums, first_solve=first_solve)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _rel(df, col="timestamp"):
    return df[col].values - df[col].values[0]

def _trim(cmd, act):
    if cmd is None or act is None:
        return cmd, act
    t_end = min(cmd["timestamp"].values[-1], act["timestamp"].values[-1])
    return (cmd[cmd["timestamp"] <= t_end].reset_index(drop=True),
            act[act["timestamp"] <= t_end].reset_index(drop=True))

def _axis_limits(solves_df, cmd, act, pad=0.3):
    pts = []
    for df in [solves_df, cmd, act]:
        if df is not None and all(c in df.columns for c in ["x","y","z"]):
            pts.append(df[["x","y","z"]].values)
    if not pts:
        return (-1,1),(-1,1),(0,2)
    xyz = np.vstack(pts)
    return ((xyz[:,0].min()-pad, xyz[:,0].max()+pad),
            (xyz[:,1].min()-pad, xyz[:,1].max()+pad),
            (max(0,xyz[:,2].min()-pad), xyz[:,2].max()+pad))

def _qmat_batch(qw, qx, qy, qz):
    """Batch unit quaternion -> rotation matrices, shape (N,3,3)."""
    q = np.stack([qw,qx,qy,qz],-1).astype(float)
    q /= np.maximum(np.linalg.norm(q,axis=-1,keepdims=True), 1e-12)
    w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
    N = len(w); R = np.zeros((N,3,3))
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-w*z); R[:,0,2]=2*(x*z+w*y)
    R[:,1,0]=2*(x*y+w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-w*x)
    R[:,2,0]=2*(x*z-w*y);   R[:,2,1]=2*(y*z+w*x);   R[:,2,2]=1-2*(x*x+y*y)
    return R

def _style3d(ax, title=""):
    """Style 3-D axes.
    NOTE: do NOT call set_rasterization_zorder here — that rasterizes all 3D
    content. Leave it out so lines stay as vector paths in the PDF."""
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#DDDDDD")
    ax.grid(True, color="#ECECEC", lw=0.4, ls="--")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(TEXT_C)
        axis.label.set_fontsize(FS - 1)
        axis.set_tick_params(colors=DIM_C, labelsize=FS - 2)
    ax.set_xlabel("x (m)", labelpad=3)
    ax.set_ylabel("y (m)", labelpad=3)
    ax.set_zlabel("z (m)", labelpad=3)
    if title:
        ax.set_title(title, fontsize=FS, color=TEXT_C, pad=5)

def _draw_body_frames(ax, xyz, quats, n_samples=10, scale=None):
    """
    Draw n_samples small RGB body-frame tripods (line segments) along xyz.
    X=red, Y=green, Z=blue. Scale is auto-derived from trajectory extent.
    """
    n_pts = len(xyz)
    if n_pts < 2:
        return
    if scale is None:
        extent = max(float(xyz[:,0].max()-xyz[:,0].min()), float(xyz[:,1].max()-xyz[:,1].min()),
                     float(xyz[:,2].max()-xyz[:,2].min()), 0.3)
        scale  = extent * 0.055
    idx = np.linspace(0, n_pts-1, n_samples, dtype=int)
    Rs  = _qmat_batch(quats[idx,0], quats[idx,1], quats[idx,2], quats[idx,3])
    for k, i in enumerate(idx):
        p = xyz[i]
        R = Rs[k]
        for ci, col in enumerate([BF_X, BF_Y, BF_Z]):
            d = R[:, ci] * scale
            ax.plot([p[0], p[0]+d[0]],
                    [p[1], p[1]+d[1]],
                    [p[2], p[2]+d[2]],
                    color=col, lw=1.6, alpha=0.90, zorder=12)


# ===============================================================================
#  Figure 1 — 3-D trajectory comparison (4 views)
# ===============================================================================

def plot_trajectories(logs: dict, solves_info: dict) -> plt.Figure:
    """
    4 orbital views at +60, +120, -60, -120 degrees azimuth.
    Shows: actual path (dashed red), commanded path (blue),
           first-solve plan (amber), and 10 sampled body-frame
           RGB tripods along the actual path. No ghost plans.
    """
    cmd   = logs["cmd"]
    act   = logs["act"]
    first = solves_info.get("first_solve")

    fig = plt.figure(figsize=(20, 16), facecolor="white")
    fig.suptitle("Figure 1 — 3-D Trajectory Comparison",
                 fontsize=FS+3, fontweight="bold", color=TEXT_C, y=0.98)

    xlim, ylim, zlim = _axis_limits(logs["solves"], cmd, act)

    xlim = (-2, 2)      # example
    ylim = (-2, 2)
    zlim = (0, 2)

    legend_handles = [
        Line2D([0],[0], color=FIRST_C, lw=2.5,            label="First-solve plan (solve 0)"),
        Line2D([0],[0], color=CMD_C,   lw=2.0,            label="Commanded path"),
        Line2D([0],[0], color=ACT_C,   lw=1.8, ls="--",   label="Actual path"),
        Line2D([0],[0], color=BF_X,    lw=1.6,            label="Body frame X"),
        Line2D([0],[0], color=BF_Y,    lw=1.6,            label="Body frame Y"),
        Line2D([0],[0], color=BF_Z,    lw=1.6,            label="Body frame Z"),
    ]

    # Pre-compute body-frame scale from actual extent (shared across views)
    bf_scale = None
    if act is not None and all(c in act.columns for c in ["x","y","z"]):
        xyz_a = act[["x","y","z"]].values
        ext   = max(float(xyz_a[:,0].max()-xyz_a[:,0].min()), float(xyz_a[:,1].max()-xyz_a[:,1].min()),
                    float(xyz_a[:,2].max()-xyz_a[:,2].min()), 0.3)
        bf_scale = ext * 0.055

    for vi, view in enumerate(VIEWS):
        ax = fig.add_subplot(2, 2, vi+1, projection="3d")
        _style3d(ax, view["label"])
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.view_init(elev=view["elev"], azim=view["azim"])

        # First-solve plan — permanent reference
        if first is not None:
            ax.plot(first["x"], first["y"], first["z"],
                    color=FIRST_C, lw=2.8, alpha=0.92, zorder=8)

        # Commanded path
        if cmd is not None:
            ax.plot(cmd["x"], cmd["y"], cmd["z"],
                    color=CMD_C, lw=2.0, zorder=10)

        # Actual path
        if act is not None:
            ax.plot(act["x"], act["y"], act["z"],
                    color=ACT_C, lw=1.8, ls="--", alpha=0.82, zorder=11)

            # 10 sampled body-frame tripods along actual path
            if all(c in act.columns for c in ["qw","qx","qy","qz"]):
                xyz_act  = act[["x","y","z"]].values
                quat_act = act[["qw","qx","qy","qz"]].values
                _draw_body_frames(ax, xyz_act, quat_act,
                                  n_samples=10, scale=bf_scale)

        if vi == 0:
            ax.legend(handles=legend_handles, fontsize=FS-2,
                      loc="upper left", framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ===============================================================================
#  Figure 2 — state tracking (one subplot per state group)
# ===============================================================================

def plot_state_tracking(logs: dict, solves_info: dict) -> plt.Figure:
    """
    2x2 grid — one subplot per state group:
      [0,0] position      x, y, z all on one axes
      [0,1] velocity      vx, vy, vz on one axes
      [1,0] quaternion    qw, qx, qy, qz on one axes
      [1,1] angular rate  wx, wy, wz on one axes

    Each axes: commanded (solid), actual (dashed), first-solve plan (dash-dot).
    """
    cmd, act = _trim(logs["cmd"], logs["act"])
    first    = solves_info.get("first_solve")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor="white")
    fig.suptitle("Figure 2 — State Tracking",
                 fontsize=FS+3, fontweight="bold", color=TEXT_C, y=0.99)
    fig.subplots_adjust(hspace=0.42, wspace=0.30,
                        top=0.935, bottom=0.07, left=0.07, right=0.97)

    groups = [
        (axes[0,0], "position",     ["x","y","z"],          "m",     "Position  x / y / z"),
        (axes[0,1], "velocity",     ["vx","vy","vz"],        "m/s",   "Velocity  vx / vy / vz"),
        (axes[1,0], "quaternion",   ["qw","qx","qy","qz"],   "—",     "Quaternion  qw / qx / qy / qz"),
        (axes[1,1], "angular_rate", ["wx","wy","wz"],        "rad/s", "Angular Rate  wx / wy / wz"),
    ]

    tc = _rel(cmd)   if cmd   is not None else None
    ta = _rel(act)   if act   is not None else None
    tf = (first["t"].values - first["t"].values[0]) if first is not None else None

    for ax, grp_key, state_vars, unit, title in groups:
        cols = GRP_COLS[grp_key]
        legend_lines = []

        for var in state_vars:
            col = cols[var]

            if cmd is not None and var in cmd.columns:
                ax.plot(tc, cmd[var].values,
                        color=col, lw=2.0, alpha=1.0)
                legend_lines.append(
                    Line2D([0],[0], color=col, lw=2.0, label=f"{var} cmd"))

            if act is not None and var in act.columns:
                ax.plot(ta, act[var].values,
                        color=col, lw=1.6, ls="--", alpha=0.68)
                legend_lines.append(
                    Line2D([0],[0], color=col, lw=1.6, ls="--", alpha=0.68,
                           label=f"{var} act"))

            if first is not None and var in first.columns:
                ax.plot(tf, first[var].values,
                        color=col, lw=1.3, ls="-.", alpha=0.45)
                legend_lines.append(
                    Line2D([0],[0], color=col, lw=1.3, ls="-.", alpha=0.45,
                           label=f"{var} plan\u2080"))

        unit_str = f"  ({unit})" if unit != "—" else ""
        ax.set_title(f"{title}{unit_str}", fontsize=FS)
        ax.set_xlabel("t  (s)", fontsize=FS-1)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        ax.tick_params(labelsize=FS-1)
        n_vars = len(state_vars)
        ax.legend(handles=legend_lines, ncol=n_vars,
                  fontsize=FS-3, loc="best", framealpha=0.88)

    return fig


# ===============================================================================
#  Error helpers
# ===============================================================================

def compute_errors(logs: dict):
    """
    Interpolate actual onto commanded timestamps, compute error = actual - commanded.
    Returns dict {var: error_array, 'time': time_array} or None.
    """
    cmd, act = _trim(logs["cmd"], logs["act"])
    if cmd is None or act is None:
        return None
    tc = cmd["timestamp"].values
    ta = act["timestamp"].values
    errors = {"time": tc - tc[0]}
    for var in ["x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz"]:
        if var in cmd.columns and var in act.columns:
            interp     = np.interp(tc, ta, act[var].values)
            errors[var] = interp - cmd[var].values
    return errors


# ===============================================================================
#  Figure 3 — tracking error time series (one subplot per group)
# ===============================================================================

def plot_tracking_errors(errors) -> plt.Figure:
    """
    2x2 grid — one subplot per state group, all error channels per group together.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor="white")
    fig.suptitle("Figure 3 — Tracking Error  (actual \u2212 commanded)",
                 fontsize=FS+3, fontweight="bold", color=TEXT_C, y=0.99)
    fig.subplots_adjust(hspace=0.42, wspace=0.30,
                        top=0.935, bottom=0.07, left=0.07, right=0.97)

    if errors is None:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "Data not available",
                    ha="center", va="center", color=DIM_C, fontsize=12,
                    transform=ax.transAxes)
        return fig

    tc = errors["time"]
    groups = [
        (axes[0,0], "position",     ["x","y","z"],         "m",     "Position error  \u0394x / \u0394y / \u0394z"),
        (axes[0,1], "velocity",     ["vx","vy","vz"],       "m/s",   "Velocity error  \u0394vx / \u0394vy / \u0394vz"),
        (axes[1,0], "quaternion",   ["qw","qx","qy","qz"],  "\u2014", "Quaternion error  \u0394qw / \u0394qx / \u0394qy / \u0394qz"),
        (axes[1,1], "angular_rate", ["wx","wy","wz"],       "rad/s", "Angular rate error  \u0394\u03c9x / \u0394\u03c9y / \u0394\u03c9z"),
    ]

    for ax, grp_key, state_vars, unit, title in groups:
        cols = GRP_COLS[grp_key]
        legend_lines = []
        for var in state_vars:
            if var not in errors:
                continue
            col = cols[var]
            err = errors[var]
            ax.plot(tc, err, color=col, lw=1.8)
            ax.fill_between(tc, err, color=col, alpha=FILL_A)
            rms = np.sqrt(np.mean(err**2))
            legend_lines.append(
                Line2D([0],[0], color=col, lw=1.8,
                       label=f"\u0394{var}  rms={rms:.3g}"))

        ax.axhline(0, color=DIM_C, lw=0.8, ls=":")
        unit_str = f"  ({unit})" if unit not in ("—","\u2014") else ""
        ax.set_title(f"{title}{unit_str}", fontsize=FS)
        ax.set_xlabel("t  (s)", fontsize=FS-1)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        ax.tick_params(labelsize=FS-1)
        ax.legend(handles=legend_lines, fontsize=FS-3, loc="best", framealpha=0.88)

    return fig


# ===============================================================================
#  Figure 4 — error box plots
# ===============================================================================

def plot_error_boxplots(errors) -> plt.Figure:
    """
    2x2 grid — one subplot per state group, one box per component.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor="white")
    fig.suptitle("Figure 4 — Error Distribution Box Plots  (actual \u2212 commanded)",
                 fontsize=FS+3, fontweight="bold", color=TEXT_C, y=0.99)
    fig.subplots_adjust(hspace=0.44, wspace=0.30,
                        top=0.930, bottom=0.08, left=0.07, right=0.97)

    if errors is None:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "Data not available",
                    ha="center", va="center", color=DIM_C, fontsize=12,
                    transform=ax.transAxes)
        return fig

    groups = [
        (axes[0,0], "position",     ["x","y","z"],         "m",     "Position errors  (m)"),
        (axes[0,1], "velocity",     ["vx","vy","vz"],       "m/s",   "Velocity errors  (m/s)"),
        (axes[1,0], "quaternion",   ["qw","qx","qy","qz"],  "",      "Quaternion errors"),
        (axes[1,1], "angular_rate", ["wx","wy","wz"],       "rad/s", "Angular rate errors  (rad/s)"),
    ]

    for ax, grp_key, state_vars, unit, title in groups:
        cols      = GRP_COLS[grp_key]
        data_list = [errors[v] for v in state_vars if v in errors]
        labels    = [v         for v in state_vars if v in errors]
        box_cols  = [cols[v]   for v in labels]

        if not data_list:
            ax.axis("off")
            continue

        bp = ax.boxplot(
            data_list, labels=labels, patch_artist=True, notch=False,
            medianprops=dict(color="white", lw=2.5),
            whiskerprops=dict(lw=1.4),
            capprops=dict(lw=1.4),
            flierprops=dict(marker="o", ms=3.0, alpha=0.40, markeredgewidth=0.4),
            widths=0.45,
        )
        for patch, col in zip(bp["boxes"], box_cols):
            patch.set_facecolor(col)
            patch.set_alpha(0.62)
        for flier, col in zip(bp["fliers"], box_cols):
            flier.set(markerfacecolor=col, markeredgecolor=col)

        ax.axhline(0, color=DIM_C, lw=0.8, ls=":")
        ax.set_title(title, fontsize=FS)
        ax.set_ylabel("Error", fontsize=FS-1)
        ax.tick_params(labelsize=FS-1)

        # Median annotation above each box
        y_top = ax.get_ylim()[1]
        for i, (var, d) in enumerate(zip(labels, data_list)):
            ax.text(i+1, y_top, f"med\n{np.median(d):.3g}",
                    ha="center", va="top", fontsize=FS-4, color=DIM_C)

    return fig


# ===============================================================================
#  Main
# ===============================================================================

def main():
    ap = argparse.ArgumentParser(description="Quadrotor MPC flight analysis — vector PDF")
    ap.add_argument("folder", help="Folder containing the CSV log files")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="Truncate logs to first N seconds")
    ap.add_argument("--out", default=None,
                    help="Output PDF path (default: <folder>/analysis_plots.pdf)")
    args = ap.parse_args()

    folder  = args.folder
    out_pdf = args.out or os.path.join(folder, "analysis_plots.pdf")

    print(f"\nLoading logs from: {folder}")
    logs = load_logs(folder, cutoff=args.cutoff)
    for key, label in [("solves","all_solves"), ("cmd","commanded"), ("act","actual")]:
        d = logs[key]
        print(f"  {label:12s}: {len(d)} rows" if d is not None else f"  {label:12s}: NOT FOUND")

    print("\nExtracting solve data ...")
    solves_info = extract_solves(logs)
    if solves_info:
        print(f"  {len(solves_info['solve_nums'])} solves found")

    print("\nBuilding figures ...")
    fig1 = plot_trajectories(logs, solves_info)
    fig2 = plot_state_tracking(logs, solves_info)
    errors = compute_errors(logs)
    fig3 = plot_tracking_errors(errors)
    fig4 = plot_error_boxplots(errors)

    print(f"\nSaving vector PDF to: {out_pdf}")
    with PdfPages(out_pdf) as pdf:
        for i, fig in enumerate([fig1, fig2, fig3, fig4], start=1):
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"  Figure {i} saved.")
        d = pdf.infodict()
        d["Title"]   = "Quadrotor MPC Flight Analysis"
        d["Subject"] = "Receding-horizon trajectory tracking report"

    print(f"\nDone — {out_pdf}")


if __name__ == "__main__":
    main()