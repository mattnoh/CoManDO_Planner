#!/usr/bin/env python3
"""
trajectory_analysis_plots.py — Receding-horizon quadrotor flight analysis
==========================================================================
Produces a single multi-page vector PDF with figures.

TWO MODES — auto-detected from available files:

  HARDWARE mode  (all three CSVs present and non‑empty):
    all_solves.csv + commanded_state.csv + actual_state.csv
    Fig 1 — 3D trajectory (4 views): first-solve plan, commanded, actual, body frames
    Fig 2 — State tracking: cmd vs actual vs plan per group
    Fig 3 — Tracking error time series (actual - commanded)
    Fig 4 — Error distribution box plots

  SIM mode  (actual_state.csv missing or empty):
    all_solves.csv + commanded_state.csv
    commanded_state.csv IS the executed path — sim_bridge feeds X[1] back as next x0,
    so the commanded states are the states the plant actually visits.
    Fig 1 — 3D trajectory (4 views): first-solve plan + executed path (= commanded)
    Fig 2 — State tracking: executed path + first-solve plan per group
    Fig 3 — Warm-start prediction error: |X[1]_prev - x0_received| over time
             (same as dxws_pred in the standalone — measures solver prediction quality)
    Fig 4 — Control trajectory: fz, Mx, My, Mz over solves
             (unique to sim — hardware uses the bridge for this)

CSV layout:
  all_solves.csv        solve_num, solve_time_ms, node,
                        t, x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz, fz,mx,my,mz
  commanded_state.csv   timestamp, x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz, fz,mx,my,mz
  actual_state.csv      timestamp, x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz

Usage:
  python3 trajectory_analysis_plots.py <folder>
  python3 trajectory_analysis_plots.py <folder> --cutoff 10 --out report.pdf
  python3 trajectory_analysis_plots.py <folder> --sim        # force sim mode
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
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning)

# ── Column names ───────────────────────────────────────────────────────────────
SOLVE_COLS = ["solve_num","solve_time_ms","node",
              "t","x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz",
              "fz","mx","my","mz"]
CMD_COLS   = ["timestamp","x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz",
              "fz","mx","my","mz"]
ACT_COLS   = ["timestamp","x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz"]

# ── Colours ────────────────────────────────────────────────────────────────────
FIRST_C = "#D97706"
CMD_C   = "#1D4ED8"
ACT_C   = "#DC2626"
ERR_C   = "#7C3AED"
BF_X    = "#EF4444"
BF_Y    = "#22C55E"
BF_Z    = "#3B82F6"
GRID_C  = "#E5E7EB"
TEXT_C  = "#111827"
DIM_C   = "#6B7280"
FILL_A  = 0.10

GRP_COLS = {
    "position":     {"x":"#1D4ED8","y":"#16A34A","z":"#DC2626"},
    "velocity":     {"vx":"#1D4ED8","vy":"#16A34A","vz":"#DC2626"},
    "quaternion":   {"qw":"#7C3AED","qx":"#1D4ED8","qy":"#16A34A","qz":"#DC2626"},
    "angular_rate": {"wx":"#1D4ED8","wy":"#16A34A","wz":"#DC2626"},
    "control":      {"fz":"#1D4ED8","mx":"#16A34A","my":"#DC2626","mz":"#7C3AED"},
}

FS = 11

VIEWS = [
    dict(elev=25, azim=60,   label="Azim +60°"),
    dict(elev=25, azim=120,  label="Azim +120°"),
    dict(elev=25, azim=-60,  label="Azim -60°"),
    dict(elev=25, azim=-120, label="Azim -120°"),
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
    "pdf.fonttype":        42,
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
            # Return None if the dataframe is empty after loading
            return raw if len(raw) > 0 else None
        except Exception as exc:
            print(f"  [warn] {fname}: {exc}")
            return None

    solves = _read("all_solves.csv",      SOLVE_COLS)
    cmd    = _read("commanded_state.csv", CMD_COLS)
    act    = _read("actual_state.csv",    ACT_COLS)
    return dict(solves=solves, cmd=cmd, act=act)


def detect_mode(logs: dict, force_sim: bool) -> str:
    if force_sim:
        return "sim"
    # If actual is None (missing or empty after loading) -> sim mode
    if logs["act"] is None:
        return "sim"
    return "hardware"


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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rel(df, col="timestamp"):
    if df is None or len(df) == 0:
        return np.array([])
    v = df[col].values
    return v - v[0]

def _trim(cmd, act):
    if cmd is None or act is None:
        return cmd, act
    t_end = min(cmd["timestamp"].values[-1], act["timestamp"].values[-1])
    return (cmd[cmd["timestamp"] <= t_end].reset_index(drop=True),
            act[act["timestamp"] <= t_end].reset_index(drop=True))

def _qmat_batch(qw, qx, qy, qz):
    q = np.stack([qw,qx,qy,qz],-1).astype(float)
    q /= np.maximum(np.linalg.norm(q,axis=-1,keepdims=True), 1e-12)
    w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
    N = len(w); R = np.zeros((N,3,3))
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-w*z); R[:,0,2]=2*(x*z+w*y)
    R[:,1,0]=2*(x*y+w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-w*x)
    R[:,2,0]=2*(x*z-w*y);   R[:,2,1]=2*(y*z+w*x);   R[:,2,2]=1-2*(x*x+y*y)
    return R

def _style3d(ax, title=""):
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#DDDDDD")
    ax.grid(True, color="#ECECEC", lw=0.4, ls="--")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(TEXT_C); axis.label.set_fontsize(FS-1)
        axis.set_tick_params(colors=DIM_C, labelsize=FS-2)
    ax.set_xlabel("x (m)", labelpad=3)
    ax.set_ylabel("y (m)", labelpad=3)
    ax.set_zlabel("z (m)", labelpad=3)
    if title:
        ax.set_title(title, fontsize=FS, color=TEXT_C, pad=5)

def _draw_body_frames(ax, xyz, quats, n_samples=10, scale=None):
    n_pts = len(xyz)
    if n_pts < 2:
        return
    if scale is None:
        # Use fixed axis range to determine scale (so body frames are consistent across views)
        # x,y in [-1.5,1.5], z in [0,1.5] -> typical extent ~3.0
        scale = 3.0 * 0.055  # ~0.165
    idx = np.linspace(0, n_pts-1, n_samples, dtype=int)
    Rs  = _qmat_batch(quats[idx,0], quats[idx,1], quats[idx,2], quats[idx,3])
    for k, i in enumerate(idx):
        p = xyz[i]; R = Rs[k]
        for ci, col in enumerate([BF_X, BF_Y, BF_Z]):
            d = R[:, ci] * scale
            ax.plot([p[0],p[0]+d[0]], [p[1],p[1]+d[1]], [p[2],p[2]+d[2]],
                    color=col, lw=1.6, alpha=0.90, zorder=12)


# ===============================================================================
#  Figure 1 — 3D trajectory (shared by both modes, labels adapt)
# ===============================================================================

def plot_trajectories(logs, solves_info, mode) -> plt.Figure:
    first = solves_info.get("first_solve")
    cmd   = logs["cmd"]
    act   = logs["act"]

    # In sim mode: commanded_state IS the executed path; no separate actual
    path_label = "Executed path (= commanded, sim)"  if mode == "sim" else "Actual path"
    path_data  = cmd                                   if mode == "sim" else act
    path_color = CMD_C                                 if mode == "sim" else ACT_C
    path_ls    = "-"                                   if mode == "sim" else "--"

    mode_tag = " [SIM MODE]" if mode == "sim" else " [HARDWARE MODE]"
    fig = plt.figure(figsize=(20, 16), facecolor="white")
    fig.suptitle(f"Figure 1 — 3-D Trajectory Comparison{mode_tag}",
                 fontsize=FS+3, fontweight="bold", color=TEXT_C, y=0.98)

    # Fixed axis limits as requested
    xlim = (-1.5, 1.5)
    ylim = (-1.5, 1.5)
    zlim = (0, 1.5)

    legend_handles = [
        Line2D([0],[0], color=FIRST_C, lw=2.5,      label="First-solve plan (solve 0)"),
        Line2D([0],[0], color=path_color, lw=2.0, ls=path_ls, label=path_label),
        Line2D([0],[0], color=BF_X,    lw=1.6,      label="Body X"),
        Line2D([0],[0], color=BF_Y,    lw=1.6,      label="Body Y"),
        Line2D([0],[0], color=BF_Z,    lw=1.6,      label="Body Z"),
    ]

    for vi, view in enumerate(VIEWS):
        ax = fig.add_subplot(2, 2, vi+1, projection="3d")
        _style3d(ax, view["label"])
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_box_aspect([1,1,1])  # equal scaling -> origin is visual centre
        ax.view_init(elev=view["elev"], azim=view["azim"])

        if first is not None and len(first) > 0:
            ax.plot(first["x"], first["y"], first["z"],
                    color=FIRST_C, lw=2.8, alpha=0.92, zorder=8)

        if path_data is not None and len(path_data) > 0:
            ax.plot(path_data["x"], path_data["y"], path_data["z"],
                    color=path_color, lw=2.0, ls=path_ls, zorder=10)
            if all(c in path_data.columns for c in ["qw","qx","qy","qz"]):
                _draw_body_frames(ax, path_data[["x","y","z"]].values,
                                  path_data[["qw","qx","qy","qz"]].values,
                                  n_samples=10)  # scale computed from fixed limits

        # In hardware mode also draw commanded path
        if mode == "hardware" and cmd is not None and len(cmd) > 0:
            ax.plot(cmd["x"], cmd["y"], cmd["z"],
                    color=CMD_C, lw=1.6, alpha=0.7, ls=":", zorder=9)
            if vi == 0:
                legend_handles.insert(2,
                    Line2D([0],[0], color=CMD_C, lw=1.6, ls=":", label="Commanded path"))

        if vi == 0:
            ax.legend(handles=legend_handles, fontsize=FS-2,
                      loc="upper left", framealpha=0.92)

    fig.tight_layout(rect=[0,0,1,0.96])
    return fig


# ===============================================================================
#  Figure 2 — State tracking
# ===============================================================================

def plot_state_tracking(logs, solves_info, mode) -> plt.Figure:
    first = solves_info.get("first_solve")
    cmd   = logs["cmd"]
    act   = logs["act"]

    # Executed data: sim = commanded, hardware = actual
    if mode == "sim":
        exec_df = cmd
        exec_label_base = "commanded"
    else:
        exec_df = act
        exec_label_base = "actual"

    plan_label = "planned"

    # ---- Time scaling: squeeze executed to plan duration ----
    scale = 1.0
    scaling_note = ""
    if first is not None and len(first) > 0 and exec_df is not None:
        t_plan = first["t"].values
        T_plan = t_plan[-1] - t_plan[0]
        t_exec = exec_df["timestamp"].values
        T_exec = t_exec[-1] - t_exec[0]
        if T_exec > 0:
            scale = T_plan / T_exec
            scaling_note = f" (executed time scaled by {scale:.2f})"
    # ---------------------------------------------------------

    mode_tag = " [SIM MODE]" if mode == "sim" else " [HARDWARE MODE]"
    fig, axes = plt.subplots(2, 2, figsize=(18,12), facecolor="white")
    fig.suptitle(f"Figure 2 — State Tracking{mode_tag}{scaling_note}",
                 fontsize=FS+3, fontweight="bold", color=TEXT_C, y=0.99)
    fig.subplots_adjust(hspace=0.42, wspace=0.30,
                        top=0.935, bottom=0.07, left=0.07, right=0.97)

    groups = [
        (axes[0,0], "position",     ["x","y","z"],         "m",     "Position  x/y/z"),
        (axes[0,1], "velocity",     ["vx","vy","vz"],       "m/s",   "Velocity  vx/vy/vz"),
        (axes[1,0], "quaternion",   ["qw","qx","qy","qz"],  "—",     "Quaternion  qw/qx/qy/qz"),
        (axes[1,1], "angular_rate", ["wx","wy","wz"],       "rad/s", "Angular rate  wx/wy/wz"),
    ]

    # Relative time for plan (always starts at 0)
    if first is not None and len(first) > 0:
        tf = first["t"].values - first["t"].values[0]
    else:
        tf = None

    # Relative time for executed (scaled)
    if exec_df is not None and len(exec_df) > 0:
        te_raw = exec_df["timestamp"].values - exec_df["timestamp"].values[0]
        te = te_raw * scale
    else:
        te = None

    for ax, grp_key, state_vars, unit, title in groups:
        cols = GRP_COLS[grp_key]
        handles = []
        labels = []

        for var in state_vars:
            col = cols[var]

            # Executed trace (solid line)
            if exec_df is not None and len(exec_df) > 0 and var in exec_df.columns:
                line, = ax.plot(te, exec_df[var].values, color=col, lw=2.0, ls='-')
                handles.append(line)
                labels.append(f"{var} {exec_label_base}")

            # Plan trace (dashed line, slightly thinner)
            if first is not None and len(first) > 0 and var in first.columns:
                line, = ax.plot(tf, first[var].values, color=col, lw=1.5, ls='--', alpha=0.7)
                handles.append(line)
                labels.append(f"{var} {plan_label}")

        unit_str = f"  ({unit})" if unit != "—" else ""
        ax.set_title(f"{title}{unit_str}")
        ax.set_xlabel("t  (s)" + (" (scaled)" if scale != 1.0 else ""), fontsize=FS-1)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        ax.tick_params(labelsize=FS-1)

        # Legend inside subplot, larger font
        if handles:
            ax.legend(handles, labels, fontsize=FS+1, loc='upper right', framealpha=0.9)

    return fig

# ===============================================================================
#  Figure 3 — hardware: tracking error | sim: warm-start prediction error
# ===============================================================================

def compute_hardware_errors(logs):
    cmd, act = _trim(logs["cmd"], logs["act"])
    if cmd is None or act is None or len(cmd) == 0 or len(act) == 0:
        return None
    tc = cmd["timestamp"].values
    ta = act["timestamp"].values
    errors = {"time": tc - tc[0]}
    for var in ["x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz"]:
        if var in cmd.columns and var in act.columns:
            errors[var] = np.interp(tc, ta, act[var].values) - cmd[var].values
    return errors


def compute_sim_prediction_errors(solves_info, cmd):
    """
    For each solve k, X[1] is the predicted next state.
    x0 at solve k+1 is what sim_bridge received = X[1] from solve k (+ noise).
    prediction error = x0_at_solve_k+1 - X[1]_from_solve_k
    This is the dxws_pred metric from the standalone.
    """
    solve_map  = solves_info.get("solve_map", {})
    solve_nums = solves_info.get("solve_nums", [])
    if len(solve_nums) < 2 or cmd is None or len(cmd) < 2:
        return None

    state_vars = ["x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz"]
    errors = {v: [] for v in state_vars}
    errors["pos_norm"] = []
    errors["vel_norm"] = []
    errors["solve_idx"] = []

    for i, sn in enumerate(solve_nums[:-1]):
        cur_solve  = solve_map[int(sn)]
        # X[1] = node index 1 of current solve
        x1_rows = cur_solve[cur_solve["node"] == 1]
        if x1_rows.empty:
            continue
        x1 = x1_rows.iloc[0]

        # x0 at next solve = commanded_state row i+1
        # (each solve emits exactly one commanded_state row = X[1])
        if i + 1 >= len(cmd):
            break
        x0_next = cmd.iloc[i + 1]

        for v in state_vars:
            if v in cur_solve.columns and v in cmd.columns:
                errors[v].append(float(x0_next[v]) - float(x1[v]))

        pos_err = np.sqrt(sum(errors[v][-1]**2 for v in ["x","y","z"]
                              if v in errors and errors[v]))
        vel_err = np.sqrt(sum(errors[v][-1]**2 for v in ["vx","vy","vz"]
                              if v in errors and errors[v]))
        errors["pos_norm"].append(pos_err)
        errors["vel_norm"].append(vel_err)
        errors["solve_idx"].append(i)

    if not errors["solve_idx"]:
        return None
    for k in errors:
        errors[k] = np.array(errors[k])
    return errors


def plot_tracking_errors_hardware(errors) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(18,12), facecolor="white")
    fig.suptitle("Figure 3 — Tracking Error  (actual − commanded)  [HARDWARE MODE]",
                 fontsize=FS+3, fontweight="bold", color=TEXT_C, y=0.99)
    fig.subplots_adjust(hspace=0.42, wspace=0.30,
                        top=0.935, bottom=0.07, left=0.07, right=0.97)

    if errors is None:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    color=DIM_C, fontsize=12, transform=ax.transAxes)
        return fig

    tc = errors["time"]
    groups = [
        (axes[0,0], "position",     ["x","y","z"],         "m",     "Position error  Δx/Δy/Δz"),
        (axes[0,1], "velocity",     ["vx","vy","vz"],       "m/s",   "Velocity error  Δvx/Δvy/Δvz"),
        (axes[1,0], "quaternion",   ["qw","qx","qy","qz"],  "—",     "Quaternion error"),
        (axes[1,1], "angular_rate", ["wx","wy","wz"],       "rad/s", "Angular rate error"),
    ]
    for ax, grp_key, state_vars, unit, title in groups:
        cols = GRP_COLS[grp_key]; handles = []
        for var in state_vars:
            if var not in errors: continue
            col = cols[var]; err = errors[var]
            ax.plot(tc, err, color=col, lw=1.8)
            ax.fill_between(tc, err, color=col, alpha=FILL_A)
            rms = np.sqrt(np.mean(err**2))
            handles.append(Line2D([0],[0], color=col, lw=1.8, label=f"Δ{var}  rms={rms:.3g}"))
        ax.axhline(0, color=DIM_C, lw=0.8, ls=":")
        unit_str = f"  ({unit})" if unit != "—" else ""
        ax.set_title(f"{title}{unit_str}")
        ax.set_xlabel("t  (s)", fontsize=FS-1)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        ax.tick_params(labelsize=FS-1)
        ax.legend(handles=handles, fontsize=FS-3, loc="best", framealpha=0.88)
    return fig


def plot_sim_prediction_errors(pred_errors) -> plt.Figure:
    """
    Fig 3 for sim: warm-start prediction quality = dxws_pred from standalone.
    Top row: per-component position and velocity prediction errors.
    Bottom: ||pos error|| and ||vel error|| norms over solve steps.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18,12), facecolor="white")
    fig.suptitle(
        "Figure 3 — Warm-Start Prediction Error  (x0_received − X[1]_predicted)  [SIM MODE]\n"
        "Same as dxws_pred in standalone — measures how accurately the solver predicted the next state",
        fontsize=FS+2, fontweight="bold", color=TEXT_C, y=0.99)
    fig.subplots_adjust(hspace=0.42, wspace=0.30,
                        top=0.920, bottom=0.07, left=0.07, right=0.97)

    if pred_errors is None:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "Not enough data\n(need ≥2 solves)",
                    ha="center", va="center", color=DIM_C, fontsize=12,
                    transform=ax.transAxes)
        return fig

    idx = pred_errors["solve_idx"]

    # [0,0] — position component errors
    ax = axes[0,0]
    for var, col in [("x","#1D4ED8"),("y","#16A34A"),("z","#DC2626")]:
        if var in pred_errors and len(pred_errors[var]):
            ax.plot(idx, pred_errors[var], color=col, lw=1.8, label=f"Δ{var}")
            ax.fill_between(idx, pred_errors[var], color=col, alpha=FILL_A)
    ax.axhline(0, color=DIM_C, lw=0.8, ls=":")
    ax.set_title("Position prediction error  (m)")
    ax.set_xlabel("Solve step"); ax.legend(fontsize=FS-3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))

    # [0,1] — velocity component errors
    ax = axes[0,1]
    for var, col in [("vx","#1D4ED8"),("vy","#16A34A"),("vz","#DC2626")]:
        if var in pred_errors and len(pred_errors[var]):
            ax.plot(idx, pred_errors[var], color=col, lw=1.8, label=f"Δ{var}")
            ax.fill_between(idx, pred_errors[var], color=col, alpha=FILL_A)
    ax.axhline(0, color=DIM_C, lw=0.8, ls=":")
    ax.set_title("Velocity prediction error  (m/s)")
    ax.set_xlabel("Solve step"); ax.legend(fontsize=FS-3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))

    # [1,0] — position norm (primary warm-start quality metric)
    ax = axes[1,0]
    pn = pred_errors["pos_norm"]
    ax.plot(idx, pn, color="#DC2626", lw=2.0)
    ax.fill_between(idx, pn, color="#DC2626", alpha=FILL_A)
    ax.axhline(0, color=DIM_C, lw=0.8, ls=":")
    rms = np.sqrt(np.mean(pn**2))
    ax.set_title(f"||Position prediction error||  rms={rms:.4g} m\n"
                 "(should stay near 0 with zero noise and perfect dynamics)")
    ax.set_xlabel("Solve step")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))

    # [1,1] — velocity norm
    ax = axes[1,1]
    vn = pred_errors["vel_norm"]
    ax.plot(idx, vn, color="#1D4ED8", lw=2.0)
    ax.fill_between(idx, vn, color="#1D4ED8", alpha=FILL_A)
    ax.axhline(0, color=DIM_C, lw=0.8, ls=":")
    rms = np.sqrt(np.mean(vn**2))
    ax.set_title(f"||Velocity prediction error||  rms={rms:.4g} m/s")
    ax.set_xlabel("Solve step")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))

    return fig


# ===============================================================================
#  Figure 4 — hardware: error box plots | sim: control trajectory
# ===============================================================================

def plot_error_boxplots(errors) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(18,12), facecolor="white")
    fig.suptitle("Figure 4 — Error Distribution Box Plots  (actual − commanded)  [HARDWARE MODE]",
                 fontsize=FS+3, fontweight="bold", color=TEXT_C, y=0.99)
    fig.subplots_adjust(hspace=0.44, wspace=0.30,
                        top=0.930, bottom=0.08, left=0.07, right=0.97)

    if errors is None:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    color=DIM_C, fontsize=12, transform=ax.transAxes)
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
            ax.axis("off"); continue
        bp = ax.boxplot(data_list, labels=labels, patch_artist=True, notch=False,
                        medianprops=dict(color="white",lw=2.5),
                        whiskerprops=dict(lw=1.4), capprops=dict(lw=1.4),
                        flierprops=dict(marker="o",ms=3.0,alpha=0.40,markeredgewidth=0.4),
                        widths=0.45)
        for patch, col in zip(bp["boxes"], box_cols):
            patch.set_facecolor(col); patch.set_alpha(0.62)
        for flier, col in zip(bp["fliers"], box_cols):
            flier.set(markerfacecolor=col, markeredgecolor=col)
        ax.axhline(0, color=DIM_C, lw=0.8, ls=":")
        ax.set_title(title); ax.set_ylabel("Error", fontsize=FS-1)
        ax.tick_params(labelsize=FS-1)
        y_top = ax.get_ylim()[1]
        for i, (var, d) in enumerate(zip(labels, data_list)):
            ax.text(i+1, y_top, f"med\n{np.median(d):.3g}",
                    ha="center", va="top", fontsize=FS-4, color=DIM_C)
    return fig


def plot_sim_controls(logs, solves_info) -> plt.Figure:
    """
    Fig 4 for sim: control inputs over solve steps.
    commanded_state.csv includes fz, mx, my, mz (U[0] from each solve).
    Also shows solve time from all_solves.csv.
    """
    cmd    = logs["cmd"]
    solves = logs["solves"]

    fig, axes = plt.subplots(2, 2, figsize=(18,12), facecolor="white")
    fig.suptitle("Figure 4 — Control Inputs & Solve Times  [SIM MODE]",
                 fontsize=FS+3, fontweight="bold", color=TEXT_C, y=0.99)
    fig.subplots_adjust(hspace=0.42, wspace=0.30,
                        top=0.935, bottom=0.07, left=0.07, right=0.97)

    t_cmd = np.arange(len(cmd)) if cmd is not None else None

    # [0,0] — thrust fz
    ax = axes[0,0]
    if cmd is not None and len(cmd) > 0 and "fz" in cmd.columns:
        ax.plot(t_cmd, cmd["fz"].values, color="#1D4ED8", lw=2.0)
        ax.axhline(0.027*9.81, color=DIM_C, lw=1.0, ls="--", label="hover thrust")
        ax.set_title("Collective thrust fz  (N)")
        ax.set_xlabel("Solve step"); ax.legend(fontsize=FS-3)
    else:
        ax.text(0.5, 0.5, "fz not in commanded_state.csv",
                ha="center", va="center", color=DIM_C, transform=ax.transAxes)

    # [0,1] — moments mx, my, mz
    ax = axes[0,1]
    if cmd is not None and len(cmd) > 0:
        for var, col in [("mx","#1D4ED8"),("my","#16A34A"),("mz","#DC2626")]:
            if var in cmd.columns:
                ax.plot(t_cmd, cmd[var].values, color=col, lw=1.8, label=var)
        ax.axhline(0, color=DIM_C, lw=0.8, ls=":")
        ax.set_title("Body moments Mx/My/Mz  (scaled)")
        ax.set_xlabel("Solve step"); ax.legend(fontsize=FS-3)
    else:
        ax.text(0.5, 0.5, "No commanded data",
                ha="center", va="center", color=DIM_C, transform=ax.transAxes)

    # [1,0] — solve time per solve
    ax = axes[1,0]
    if solves is not None and len(solves) > 0 and "solve_time_ms" in solves.columns:
        # One solve time per solve_num — take the value at node=0
        st = solves[solves["node"] == 0][["solve_num","solve_time_ms"]].drop_duplicates()
        ax.bar(st["solve_num"].values, st["solve_time_ms"].values,
               color="#7C3AED", alpha=0.7, width=0.6)
        ax.axhline(st["solve_time_ms"].mean(), color=DIM_C, lw=1.2, ls="--",
                   label=f"mean={st['solve_time_ms'].mean():.1f} ms")
        ax.set_title("Solve time per solve  (ms)")
        ax.set_xlabel("Solve number"); ax.legend(fontsize=FS-3)
    else:
        ax.text(0.5, 0.5, "solve_time_ms not found",
                ha="center", va="center", color=DIM_C, transform=ax.transAxes)

    # [1,1] — executed z height over solve steps (quick landing check)
    ax = axes[1,1]
    if cmd is not None and len(cmd) > 0 and "z" in cmd.columns:
        ax.plot(t_cmd, cmd["z"].values, color="#DC2626", lw=2.0, label="z (m)")
        if "x" in cmd.columns:
            ax.plot(t_cmd, cmd["x"].values, color="#1D4ED8", lw=1.6, ls="--", label="x (m)")
        if "y" in cmd.columns:
            ax.plot(t_cmd, cmd["y"].values, color="#16A34A", lw=1.6, ls="--", label="y (m)")
        ax.axhline(0.1, color=DIM_C, lw=1.0, ls=":", label="z_ref=0.1 m")
        ax.set_title("Executed position over solve steps")
        ax.set_xlabel("Solve step"); ax.legend(fontsize=FS-3)
    else:
        ax.text(0.5, 0.5, "No position data",
                ha="center", va="center", color=DIM_C, transform=ax.transAxes)

    return fig


# ===============================================================================
#  Main
# ===============================================================================

def main():
    ap = argparse.ArgumentParser(description="Quadrotor MPC flight analysis — vector PDF")
    ap.add_argument("folder", help="Folder containing the CSV log files")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="Truncate logs to first N seconds (e.g., 10 for a 10‑second window)")
    ap.add_argument("--sim", action="store_true",
                    help="Force sim mode (auto-detected if actual_state.csv is absent or empty)")
    ap.add_argument("--out", default=None,
                    help="Output PDF path (default: <folder>/analysis_plots.pdf)")
    args = ap.parse_args()

    folder  = args.folder
    out_pdf = args.out or os.path.join(folder, "analysis_plots.pdf")

    print(f"\nLoading logs from: {folder}")
    logs = load_logs(folder, cutoff=args.cutoff)
    for key, label in [("solves","all_solves"), ("cmd","commanded"), ("act","actual")]:
        d = logs[key]
        if d is not None:
            print(f"  {label:12s}: {len(d)} rows")
        else:
            print(f"  {label:12s}: NOT FOUND")

    mode = detect_mode(logs, force_sim=args.sim)
    print(f"\n  Mode: {mode.upper()}")
    if mode == "sim":
        print("  (actual_state.csv missing or empty — using commanded_state.csv as executed path)")
        print("  Fig 3 = warm-start prediction error (dxws_pred)")
        print("  Fig 4 = control inputs & solve times")
    else:
        print("  Fig 3 = tracking error (actual - commanded)")
        print("  Fig 4 = error box plots")

    print("\nExtracting solve data ...")
    solves_info = extract_solves(logs)
    if solves_info and solves_info.get("solve_nums") is not None:
        print(f"  {len(solves_info['solve_nums'])} solves found")

    print("\nBuilding figures ...")
    fig1 = plot_trajectories(logs, solves_info, mode)
    fig2 = plot_state_tracking(logs, solves_info, mode)

    if mode == "hardware":
        hw_errors = compute_hardware_errors(logs)
        fig3 = plot_tracking_errors_hardware(hw_errors)
        fig4 = plot_error_boxplots(hw_errors)
    else:
        pred_errors = compute_sim_prediction_errors(solves_info, logs["cmd"])
        fig3 = plot_sim_prediction_errors(pred_errors)
        fig4 = plot_sim_controls(logs, solves_info)

    print(f"\nSaving vector PDF to: {out_pdf}")
    with PdfPages(out_pdf) as pdf:
        for i, fig in enumerate([fig1, fig2, fig3, fig4], start=1):
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"  Figure {i} saved.")
        d = pdf.infodict()
        d["Title"]   = "Quadrotor MPC Flight Analysis"
        d["Subject"] = f"Mode: {mode}"

    print(f"\nDone — {out_pdf}")


if __name__ == "__main__":
    main()