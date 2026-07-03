"""One-file landing animator for ALIPDDP benchmark results.

This is the single entry point for the quad landing visualizations:

    python3 benchmark/animate.py --dir benchmark/results/quad_cf_tracking_rh_landing_stateswitch_bcone --out rh.png
    python3 benchmark/animate.py --dir benchmark/results/quad_cf_targetframe --kind tf --out tf.mp4
    python3 benchmark/animate.py --dir benchmark/results/quad_cf_bodyframe --kind bf --out bf.gif

PNG/PDF/SVG output writes a paper-style snapshot summary. MP4/GIF output writes
an animation. No other landing animation modules are required.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


# Paper-style visual language from the reference landing plot.
mpl.rcParams["figure.dpi"] = 120
mpl.rc("font", family="serif", serif=["Computer Modern Roman", "DejaVu Serif"])
mpl.rcParams.update({
    "figure.autolayout": False,
    "font.size": 12,
    "legend.fontsize": 9,
    "axes.xmargin": 0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.dash_capstyle": "round",
    "lines.dash_joinstyle": "round",
})

BG = "#ffffff"
PANE = "#ffffff"
PANEL = "#ffffff"
GRID = "#d8d8d8"
TEXT = "#111111"
GROUND = "#e8f1e8"
DRONE = "#1f77b4"
TARGET = "#d62728"
PLAN_D = "#1f77b4"
PLAN_T = "#ff7f0e"
PRED_T = "#17a65b"
CONE = "#2ca02c"
VIOL = "#ff7f0e"
PHASE = "#6f42c1"
INACTIVE = "#9aa0a6"
THRUST = "#8c564b"
SOLVE = "#7f7f7f"
BX = "#d62728"
BY = "#2ca02c"
BZ = "#1f77b4"
ARM = "#3f4f5f"
RING = "#63788c"
WHEEL = "#202020"
CMAP = "rainbow"

ARM_LEN = 0.40
MOTOR_R = 0.28
N_RING = 80
RING_T = np.linspace(0, 2 * np.pi, N_RING)
ARM_ANGLES = np.linspace(np.pi / 4, np.pi / 4 + 2 * np.pi, 4, endpoint=False)
TARGET_HALF = 0.35
TARGET_LIFT = 0.012
WHEEL_RADIUS = 0.065
WHEEL_Z_OFFSET = -0.075
GS_DEG = 35.0
GS_TAN = np.tan(np.radians(GS_DEG))
GS_APEX_OFFSET = 0.1
BCONE_HALF_ANGLE_DEG = 35.0
BCONE_TAN = np.tan(np.radians(BCONE_HALF_ANGLE_DEG))
BCONE_OFFSET = 0.5
CONE_H = 3.5
CONE_N = 16
CONE_ALPHA = 0.50
PHASE_ON_THRESHOLD = 0.95
PHASE_VIS_THRESHOLD = 0.05

# Animation layout knobs.
ANIM_FIGSIZE = (18, 24)
ANIM_SAVE_DPI = 140
ANIM_HEIGHT_RATIOS = (0.90, 1.55)  # top 3D row, bottom 3x3 telemetry row
ANIM_GRADIENT_WIDTH = 0.018
ANIM_3D_WIDTH = 0.90

# Fixed 3D view limits. Set ANIM_USE_FIXED_3D_LIMITS = False to auto-fit.
ANIM_USE_FIXED_3D_LIMITS = True
ANIM_X_LIM = (-6.0, 6.0)
ANIM_Y_LIM = (-6.0, 6.0)
ANIM_Z_LIM = (0.0, 3.0)


def _normalize_sh_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename single-horizon CSV columns to the RH executed-CSV schema.

    The single-horizon CSV uses rx/ry/rz, t_acc, trig_env etc. while the
    rest of animate.py expects x/y/z, t, trig etc.  This function adds the
    aliased columns without removing the originals so existing callers that
    happen to reference the original names (e.g. solve_groups) still work.
    """
    df = df.copy()
    renames = {
        "t_acc":      "t",
        "rx":         "x",  "ry":  "y",  "rz":  "z",
        "rvx":        "vx", "rvy": "vy", "rvz": "vz",
        "trig_env":   "trig",
        "sigma_land": "sigma",
        "rel_speed":  "stc_rel_speed",
        "tilt_sin":   "stc_tilt_sin",
        "omega_norm": "stc_omega_norm",
    }
    for src, dst in renames.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    # Stationary target: absolute position = relative position (target at origin)
    for c in ("tx", "ty", "tz", "tvx", "tvy", "tvz"):
        if c not in df.columns:
            df[c] = 0.0
    if "rh" not in df.columns:
        df["rh"] = 0
    if "dist" not in df.columns:
        df["dist"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    if "trigger_active" not in df.columns:
        trig = df["trig"].values if "trig" in df.columns else np.zeros(len(df))
        df["trigger_active"] = (trig > 1e-6).astype(float)
    if "stc_inc" not in df.columns:
        df["stc_inc"] = 0.0
    if "phase" not in df.columns:
        df["phase"] = df["sigma"].values if "sigma" in df.columns else 0.0
    return df


def col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name in df.columns:
        vals = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        return np.nan_to_num(vals, nan=default, posinf=default, neginf=default)
    return np.full(len(df), default, dtype=float)


def stack_cols(df: pd.DataFrame, names: tuple[str, str, str], default: float = 0.0) -> np.ndarray:
    return np.stack([col(df, names[0], default), col(df, names[1], default), col(df, names[2], default)], axis=1)


def quat_to_rotmat(q) -> np.ndarray:
    qw, qx, qy, qz = q
    n = max(float(np.linalg.norm(q)), 1e-9)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def quat_mul(q1, q2) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def seg3d(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def cone_axes(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = v / max(np.linalg.norm(v), 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(v, ref)
    u = u / max(np.linalg.norm(u), 1e-9)
    return u, np.cross(v, u)


def cone_wire(apex, axis, tan_cone, height, rim_theta, spoke_theta):
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    u, v = cone_axes(axis)
    center = apex + height * axis
    radius = tan_cone * height
    rim = center[None, :] + radius * (
        np.cos(rim_theta)[:, None] * u[None, :] + np.sin(rim_theta)[:, None] * v[None, :]
    )
    spokes = []
    for th in spoke_theta:
        p = center + radius * (np.cos(th) * u + np.sin(th) * v)
        spokes.append(np.vstack([apex, p]))
    return rim, spokes


def target_pad_vertices(center, yaw=0.0, half=TARGET_HALF):
    c, s = np.cos(yaw), np.sin(yaw)
    bx = np.array([c, s, 0.0])
    by = np.array([-s, c, 0.0])
    o = np.asarray(center, dtype=float)
    return [o - half * bx - half * by, o + half * bx - half * by,
            o + half * bx + half * by, o - half * bx + half * by]


def target_pad_points(center, yaw=0.0, half=TARGET_HALF, lift=TARGET_LIFT):
    c, s = np.cos(yaw), np.sin(yaw)
    bx = np.array([c, s, 0.0])
    by = np.array([-s, c, 0.0])
    o = np.asarray(center, dtype=float) + np.array([0.0, 0.0, lift])
    corners = np.array([
        o - half * bx - half * by,
        o + half * bx - half * by,
        o + half * bx + half * by,
        o - half * bx + half * by,
        o - half * bx - half * by,
    ])
    cross_x = np.array([o - 0.55 * half * bx, o + 0.55 * half * bx])
    cross_y = np.array([o - 0.55 * half * by, o + 0.55 * half * by])
    heading = np.array([o + 0.18 * half * bx, o + 1.28 * half * bx])
    return corners, cross_x, cross_y, heading


def target_wheel_points(center, yaw=0.0, half=TARGET_HALF):
    c, s = np.cos(yaw), np.sin(yaw)
    bx = np.array([c, s, 0.0])
    by = np.array([-s, c, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    base = np.asarray(center, dtype=float) + np.array([0.0, 0.0, WHEEL_Z_OFFSET])
    theta = np.linspace(0, 2 * np.pi, 24)
    wheels = []
    for sx in (-0.58, 0.58):
        for sy in (-1.08, 1.08):
            wc = base + sx * half * bx + sy * half * by
            wheels.append(wc[None, :] + WHEEL_RADIUS * np.cos(theta)[:, None] * bx[None, :]
                          + WHEEL_RADIUS * np.sin(theta)[:, None] * ez[None, :])
    return wheels


def make_target_pad_artists(ax):
    pad = Poly3DCollection([target_pad_vertices(np.zeros(3))], color=TARGET, alpha=0.62, zorder=10)
    ax.add_collection3d(pad)
    outline, = ax.plot([], [], [], color=TARGET, lw=2.0, alpha=0.95, zorder=11)
    cross_x, = ax.plot([], [], [], color=TEXT, lw=1.2, alpha=0.75, zorder=12)
    cross_y, = ax.plot([], [], [], color=TEXT, lw=1.2, alpha=0.75, zorder=12)
    heading, = ax.plot([], [], [], color=TARGET, lw=3.0, alpha=0.95, zorder=12)
    wheels = [ax.plot([], [], [], color=WHEEL, lw=3.0, alpha=0.88, zorder=9)[0] for _ in range(4)]
    return {"pad": pad, "outline": outline, "cross_x": cross_x,
            "cross_y": cross_y, "heading": heading, "wheels": wheels}


def update_target_pad(artists, center, yaw=0.0):
    artists["pad"].set_verts([target_pad_vertices(center, yaw)])
    corners, cross_x, cross_y, heading = target_pad_points(center, yaw)
    artists["outline"].set_data_3d(corners[:, 0], corners[:, 1], corners[:, 2])
    artists["cross_x"].set_data_3d(cross_x[:, 0], cross_x[:, 1], cross_x[:, 2])
    artists["cross_y"].set_data_3d(cross_y[:, 0], cross_y[:, 1], cross_y[:, 2])
    artists["heading"].set_data_3d(heading[:, 0], heading[:, 1], heading[:, 2])
    for wheel, pts in zip(artists["wheels"], target_wheel_points(center, yaw)):
        wheel.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])


def target_pad_artist_list(artists):
    return [artists["pad"], artists["outline"], artists["cross_x"], artists["cross_y"],
            artists["heading"], *artists["wheels"]]


def draw_target_pad(ax, center, yaw=0.0, alpha=0.72):
    artists = make_target_pad_artists(ax)
    artists["pad"].set_alpha(alpha)
    update_target_pad(artists, center, yaw)


def draw_drone(ax, pos, quat, scale=0.75, alpha=0.9):
    pos = np.asarray(pos, dtype=float)
    r = quat_to_rotmat(quat)
    bx = r[:, 0]
    by = r[:, 1]
    for th in ARM_ANGLES:
        tip = pos + r @ (np.array([np.cos(th), np.sin(th), 0.0]) * ARM_LEN * scale)
        ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]],
                color=ARM, lw=2.0, alpha=alpha, solid_capstyle="round")
        ring = tip[None, :] + MOTOR_R * scale * (
            np.cos(RING_T)[:, None] * bx[None, :] + np.sin(RING_T)[:, None] * by[None, :]
        )
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=RING, lw=0.9, alpha=alpha)
    hs = ARM_LEN * scale * 0.20
    hx0 = pos + r @ np.array([-hs, 0, 0])
    hx1 = pos + r @ np.array([hs, 0, 0])
    hy0 = pos + r @ np.array([0, -hs, 0])
    hy1 = pos + r @ np.array([0, hs, 0])
    ax.plot([hx0[0], hx1[0]], [hx0[1], hx1[1]], [hx0[2], hx1[2]], color=ARM, lw=2.0, alpha=alpha)
    ax.plot([hy0[0], hy1[0]], [hy0[1], hy1[1]], [hy0[2], hy1[2]], color=ARM, lw=2.0, alpha=alpha)
    axis_len = 0.55 * scale
    for color, axis in ((BX, 0), (BY, 1), (BZ, 2)):
        end = pos + axis_len * r[:, axis]
        ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]], color=color, lw=1.3, alpha=alpha)


def make_drone_artists(ax):
    artists = {
        "arms": [ax.plot([], [], [], color=ARM, lw=3.0, solid_capstyle="round", zorder=8)[0]
                 for _ in ARM_ANGLES],
        "rings": [ax.plot([], [], [], color=RING, lw=1.2, alpha=0.85, zorder=9)[0]
                  for _ in ARM_ANGLES],
    }
    artists["hub_x"], = ax.plot([], [], [], color=ARM, lw=2.6, zorder=10)
    artists["hub_y"], = ax.plot([], [], [], color=ARM, lw=2.6, zorder=10)
    return artists


def update_drone(artists, pos, quat, scale=1.0):
    pos = np.asarray(pos, dtype=float)
    r = quat_to_rotmat(quat)
    bx = r[:, 0]
    by = r[:, 1]
    for k, th in enumerate(ARM_ANGLES):
        tip = pos + r @ (np.array([np.cos(th), np.sin(th), 0.0]) * ARM_LEN * scale)
        artists["arms"][k].set_data_3d([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]])
        ring = tip[None, :] + MOTOR_R * scale * (
            np.cos(RING_T)[:, None] * bx[None, :] + np.sin(RING_T)[:, None] * by[None, :]
        )
        artists["rings"][k].set_data_3d(ring[:, 0], ring[:, 1], ring[:, 2])
    hs = ARM_LEN * scale * 0.20
    hx0 = pos + r @ np.array([-hs, 0, 0])
    hx1 = pos + r @ np.array([hs, 0, 0])
    hy0 = pos + r @ np.array([0, -hs, 0])
    hy1 = pos + r @ np.array([0, hs, 0])
    artists["hub_x"].set_data_3d([hx0[0], hx1[0]], [hx0[1], hx1[1]], [hx0[2], hx1[2]])
    artists["hub_y"].set_data_3d([hy0[0], hy1[0]], [hy0[1], hy1[1]], [hy0[2], hy1[2]])


def drone_artist_list(artists):
    return [*artists["arms"], *artists["rings"], artists["hub_x"], artists["hub_y"]]


def style_3d(ax, title):
    ax.set_facecolor(BG)
    ax.grid(False)
    for attr in ("xaxis", "yaxis", "zaxis"):
        pane = getattr(ax, attr).pane
        pane.fill = True
        pane.set_facecolor(PANE)
        pane.set_edgecolor(GRID)
        getattr(ax, attr)._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.tick_params(colors=TEXT, labelsize=8)
    for label in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        label.set_color(TEXT)
        label.set_fontsize(9)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)


def style_panel(ax, title, ylabel=None):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.grid(True, color=GRID, lw=0.45, ls="--", alpha=0.55)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=TEXT)
    ax.set_xlabel("Time [s]", fontsize=8, color=TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)


def detect_kind(base: Path, explicit: str | None = None) -> str:
    if explicit and explicit != "auto":
        return {"tf": "target", "target-frame": "target", "target": "target",
                "bf": "body", "body-frame": "body", "body": "body",
                "rh": "rh", "rh-shifted": "rh",
                "sh": "sh", "single-horizon": "sh", "single_horizon": "sh"}.get(explicit, explicit)
    lower = str(base).lower().replace("-", "_").replace("/", "_")
    if (base / "single_horizon_stc.csv").exists():
        return "sh"
    if (base / "moving_executed.csv").exists() and (base / "moving_all_solves.csv").exists():
        return "rh"
    if (base / "executed.csv").exists() and (base / "all_solves.csv").exists():
        cols = set(pd.read_csv(base / "executed.csv", nrows=0).columns)
        if {"drone_x", "drone_y", "drone_z", "p0", "p1", "p2"}.issubset(cols):
            if "bodyframe" in lower or "_bf" in lower or "body_frame" in lower:
                return "body"
            if "targetframe" in lower or "_tf" in lower or "target_frame" in lower:
                return "target"
            return "target"
        if {"x", "y", "z", "tx", "ty", "tz"}.issubset(cols):
            return "rh"
    raise FileNotFoundError(f"Could not detect landing result schema in {base}")


def load_result(dir_arg: str, explicit_kind: str | None):
    base = Path(dir_arg)
    if base.is_file():
        base = base.parent
    kind = detect_kind(base, explicit_kind)
    if kind == "sh":
        csv_path = base / "single_horizon_stc.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No single_horizon_stc.csv in {base}")
        raw_df = pd.read_csv(csv_path)
        exec_df = _normalize_sh_df(raw_df)
        # Build a solves_df with the columns solve_groups("rh", ...) expects.
        solves_df = raw_df.copy()
        solves_df["solve_id"] = 0
        solves_df["solve_ms"] = 0.0
        for c in ("tgt_x", "tgt_y", "tgt_z"):
            if c not in solves_df.columns:
                solves_df[c] = 0.0
        for src, dst in [("rx", "x"), ("ry", "y"), ("rz", "z")]:
            if src in solves_df.columns and dst not in solves_df.columns:
                solves_df[dst] = solves_df[src]
        return base, kind, exec_df, solves_df
    if kind == "rh":
        exec_path = base / "moving_executed.csv"
        solves_path = base / "moving_all_solves.csv"
        if not exec_path.exists():
            exec_path = base / "executed.csv"
        if not solves_path.exists():
            solves_path = base / "all_solves.csv"
    else:
        exec_path = base / "executed.csv"
        solves_path = base / "all_solves.csv"
    exec_df = pd.read_csv(exec_path)
    solves_df = pd.read_csv(solves_path)
    return base, kind, exec_df, solves_df


def load_constraints(base: Path) -> dict:
    path = base / "constraints.csv"
    if not path.exists():
        return {}
    result: dict = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                result[row["param"]] = float(row["value"])
            except ValueError:
                result[row["param"]] = row["value"]
    return result


def _parse_cpp_scalar(text: str, name: str) -> float | None:
    pattern = rf"\b{name}\b\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def infer_single_horizon_stc_constraints() -> dict:
    """Read single-horizon STC plot limits from the matching C++ source.

    Older single-horizon outputs only contain single_horizon_stc.csv, so the
    animator cannot get cone parameters from constraints.csv. Keeping this
    source-derived fallback here means changing GS_CONE_HALF_ANGLE_DEG or
    GS_APEX_OFFSET in the example automatically updates the visualization.
    """
    source = (Path(__file__).resolve().parents[1] /
              "problem_examples/landing/quad_cf_landing_single_horizon_stc.cpp")
    try:
        text = source.read_text()
    except OSError:
        return {}

    result: dict = {"mode": "gs", "variant": "single_horizon_stc"}
    gs_cone_deg = _parse_cpp_scalar(text, "GS_CONE_HALF_ANGLE_DEG")
    if gs_cone_deg is not None:
        result["gs_cone_half_angle_deg"] = gs_cone_deg
        result["gs_cone_tan"] = float(np.tan(np.radians(gs_cone_deg)))
    else:
        gs_stc_deg = _parse_cpp_scalar(text, "GS_STC_DEG")
        if gs_stc_deg is not None:
            result["gs_stc_deg"] = gs_stc_deg
            result["gs_stc_tan"] = float(np.tan(np.radians(gs_stc_deg)))
    gs_apex_offset = _parse_cpp_scalar(text, "GS_APEX_OFFSET")
    if gs_apex_offset is not None:
        result["gs_apex_offset"] = gs_apex_offset
    return result


def build_data(kind: str, exec_df: pd.DataFrame, solves_df: pd.DataFrame, mode: str):
    if kind in ("rh", "sh"):
        dpos = stack_cols(exec_df, ("x", "y", "z"))
        tpos = stack_cols(exec_df, ("tx", "ty", "tz"))
        q = np.stack([col(exec_df, "qw", 1.0), col(exec_df, "qx"), col(exec_df, "qy"), col(exec_df, "qz")], axis=1)
        time = col(exec_df, "t")
        rh = col(exec_df, "rh").astype(int)
        local = dpos - tpos
        dvel = stack_cols(exec_df, ("vx", "vy", "vz"))
        tvel = stack_cols(exec_df, ("tvx", "tvy", "tvz"))
        local_v = dvel - tvel
        dist = col(exec_df, "dist", np.nan)
        yaw = np.where(np.linalg.norm(tvel[:, :2], axis=1) > 1e-8, np.arctan2(tvel[:, 1], tvel[:, 0]), 0.0)
    else:
        dpos = stack_cols(exec_df, ("drone_x", "drone_y", "drone_z"))
        tpos = stack_cols(exec_df, ("tgt_x", "tgt_y", "tgt_z"))
        q_nb = np.stack([col(exec_df, "qw", 1.0), col(exec_df, "qx"), col(exec_df, "qy"), col(exec_df, "qz")], axis=1)
        q_wn = np.stack([col(exec_df, "tgt_qw", 1.0), col(exec_df, "tgt_qx"), col(exec_df, "tgt_qy"), col(exec_df, "tgt_qz")], axis=1)
        q = q_nb.copy() if kind == "body" else np.array([quat_mul(q_wn[i], q_nb[i]) for i in range(len(q_nb))])
        target_axis_x = np.array([quat_to_rotmat(q_wn[i])[:, 0] for i in range(len(q_wn))])
        time = col(exec_df, "t")
        rh = col(exec_df, "rh").astype(int)
        local = stack_cols(exec_df, ("p0", "p1", "p2"))
        local_v = stack_cols(exec_df, ("v0", "v1", "v2"))
        dist = col(exec_df, "p_err", np.nan)
        yaw = np.zeros(len(exec_df))
        for i in range(len(exec_df)):
            rwn = quat_to_rotmat(q_wn[i])
            yaw[i] = np.arctan2(rwn[1, 0], rwn[0, 0])

    if np.any(np.isnan(time)) or np.any(np.diff(time) <= 0):
        time = np.arange(len(exec_df), dtype=float) * 0.05
    abs_vel = np.gradient(dpos, time, axis=0) if len(time) > 2 else np.zeros_like(dpos)
    abs_vel = np.nan_to_num(abs_vel)
    speed = np.linalg.norm(abs_vel, axis=1)
    rel = dpos - tpos
    dxy = np.linalg.norm(rel[:, :2], axis=1)
    dz = rel[:, 2] + GS_APEX_OFFSET
    gs_margin = GS_TAN * np.maximum(0.0, dz) - dxy
    v_hat = None
    if mode == "bcone":
        if kind == "rh":
            tv = stack_cols(exec_df, ("tvx", "tvy", "tvz"))
            vm = np.linalg.norm(tv, axis=1).clip(1e-9)
            v_hat = tv / vm[:, None]
        else:
            v_hat = target_axis_x.copy()
        along = np.sum(rel * v_hat, axis=1)
        lat = np.linalg.norm(rel - along[:, None] * v_hat, axis=1)
        margin = BCONE_TAN * (BCONE_OFFSET - along) - lat
    else:
        margin = gs_margin
    phase = col(exec_df, "capture_phase", col(exec_df, "phase", 0.0))
    if "alt_trigger" in exec_df.columns:
        alt_trigger = col(exec_df, "alt_trigger", 0.0)
    elif "trig" in exec_df.columns:
        alt_trigger = col(exec_df, "trig", 0.0)
    else:
        alt_trigger = np.zeros(len(exec_df))
    trigger_active = col(exec_df, "trigger_active", (alt_trigger > 1e-6).astype(float))
    stc_inc = col(exec_df, "stc_inc", 0.0)
    ctcs_density = col(exec_df, "ctcs_density", 0.0)
    stc_acc = col(exec_df, "stc_acc", 0.0)
    gs_violation = col(exec_df, "gs_violation", np.maximum(0.0, -gs_margin))
    vz_violation = col(exec_df, "vz_violation", 0.0)
    tilt_sin_fallback = np.array([
        np.sqrt(quat_to_rotmat(qi)[0, 2] ** 2 + quat_to_rotmat(qi)[1, 2] ** 2)
        for qi in q
    ])
    stc_rel_speed = (col(exec_df, "stc_rel_speed")
                     if "stc_rel_speed" in exec_df.columns
                     else np.linalg.norm(local_v, axis=1))
    stc_tilt_sin = (col(exec_df, "stc_tilt_sin")
                    if "stc_tilt_sin" in exec_df.columns
                    else tilt_sin_fallback)
    stc_tilt_deg = np.degrees(np.arcsin(np.clip(stc_tilt_sin, 0.0, 1.0)))
    stc_tilt_qxy = np.column_stack([q[:, 1], q[:, 2]]) if len(q) else np.zeros((0, 2))
    if "stc_omega_norm" in exec_df.columns:
        stc_omega_norm = col(exec_df, "stc_omega_norm")
    elif all(c in exec_df.columns for c in ("wx", "wy", "wz")):
        stc_omega_norm = np.linalg.norm(stack_cols(exec_df, ("wx", "wy", "wz")), axis=1)
    else:
        stc_omega_norm = np.zeros(len(exec_df))
    stc_fz_b = col(exec_df, "stc_fz_B", col(exec_df, "fz_B", 0.0))
    stc_speed_violation = col(exec_df, "stc_speed_violation", 0.0)
    stc_tilt_violation = col(exec_df, "stc_tilt_violation", 0.0)
    stc_omega_violation = col(exec_df, "stc_omega_violation", 0.0)
    stc_thrust_hi_violation = col(exec_df, "stc_thrust_hi_violation",
                                  col(exec_df, "land_fmax_violation", 0.0))
    stc_thrust_lo_violation = col(exec_df, "stc_thrust_lo_violation", 0.0)
    land_fmax_violation = col(exec_df, "land_fmax_violation", 0.0)
    land_fmax_active = col(exec_df, "land_fmax_active", 0.0) > 0.5
    has_ctcs = ("ctcs_a_altitude_state" in exec_df.columns) or ("ctcs_a_env" in exec_df.columns)
    ctcs_y = col(exec_df, "ctcs_y", 0.0)
    ctcs_integrand = col(exec_df, "ctcs_integrand", col(exec_df, "ctcs_density", 0.0))
    ctcs_a_altitude_state = col(exec_df, "ctcs_a_altitude_state", col(exec_df, "ctcs_a_env", 0.0))
    ctcs_a_thrust = col(exec_df, "ctcs_a_thrust", 0.0)
    ctcs_altitude_state_active = ctcs_a_altitude_state > 1e-9
    ctcs_thrust_active = ctcs_a_thrust > 1e-9
    stc_active = (ctcs_altitude_state_active | ctcs_thrust_active) if has_ctcs else alt_trigger > 1e-6
    switch_idx = np.flatnonzero(stc_active)
    phase_switch_time = float(time[switch_idx[0]]) if len(switch_idx) else None
    return {
        "dpos": dpos, "tpos": tpos, "quat": q, "time": time, "rh": rh,
        "local": local, "local_v": local_v, "dist": dist, "yaw": yaw,
        "abs_vel": abs_vel, "speed": speed, "margin": margin, "v_hat": v_hat,
        "phase": phase, "alt_trigger": alt_trigger,
        "trigger_active": trigger_active, "stc_inc": stc_inc,
        "ctcs_density": ctcs_density, "stc_acc": stc_acc,
        "gs_violation": gs_violation, "vz_violation": vz_violation,
        "stc_rel_speed": stc_rel_speed,
        "stc_tilt_sin": stc_tilt_sin,
        "stc_tilt_deg": stc_tilt_deg,
        "stc_tilt_qxy": stc_tilt_qxy,
        "stc_omega_norm": stc_omega_norm,
        "stc_fz_b": stc_fz_b,
        "stc_speed_violation": stc_speed_violation,
        "stc_tilt_violation": stc_tilt_violation,
        "stc_omega_violation": stc_omega_violation,
        "stc_thrust_hi_violation": stc_thrust_hi_violation,
        "stc_thrust_lo_violation": stc_thrust_lo_violation,
        "has_ctcs": has_ctcs,
        "ctcs_y": ctcs_y,
        "ctcs_integrand": ctcs_integrand,
        "ctcs_a_altitude_state": ctcs_a_altitude_state,
        "ctcs_a_thrust": ctcs_a_thrust,
        "ctcs_altitude_state_active": ctcs_altitude_state_active,
        "ctcs_thrust_active": ctcs_thrust_active,
        "land_fmax_violation": land_fmax_violation,
        "land_fmax_active": land_fmax_active,
        "stc_active": stc_active, "phase_switch_time": phase_switch_time,
        "exec": exec_df, "solves": solves_df, "kind": kind, "mode": mode,
    }


def solve_groups(kind: str, solves_df: pd.DataFrame):
    key = "solve_id" if "solve_id" in solves_df.columns else "rh"
    groups = {}
    for sid, grp in solves_df.groupby(key):
        grp = grp.sort_values("node") if "node" in grp.columns else grp
        if kind in ("rh", "sh"):
            drone = stack_cols(grp, ("x", "y", "z"))
            target = stack_cols(grp, ("tgt_x", "tgt_y", "tgt_z"))
            pred = stack_cols(grp, ("pred_tgt_x", "pred_tgt_y", "pred_tgt_z")) if "pred_tgt_x" in grp else None
            local = stack_cols(grp, ("rx", "ry", "rz")) if "rx" in grp else drone - target
        else:
            drone = stack_cols(grp, ("drone_x", "drone_y", "drone_z")) if "drone_x" in grp else None
            target = stack_cols(grp, ("tgt_x", "tgt_y", "tgt_z")) if "tgt_x" in grp else None
            pred = stack_cols(grp, ("ptgt_x", "ptgt_y", "ptgt_z")) if "ptgt_x" in grp else None
            local = stack_cols(grp, ("p0", "p1", "p2"))
        groups[int(sid)] = {"drone": drone, "target": target, "pred": pred, "local": local}
    return groups


def solve_id_for_rh(rh_id: int, groups: dict):
    """Executed RH rows are zero-based; solve CSV includes cold solve as id 0."""
    if rh_id + 1 in groups:
        return rh_id + 1
    return rh_id


def draw_constraint(ax, mode: str, target_pos, v_axis=None, alpha=CONE_ALPHA):
    rim_theta = np.linspace(0, 2 * np.pi, 80)
    spoke_theta = np.linspace(0, 2 * np.pi, CONE_N, endpoint=False)
    if mode == "bcone" and v_axis is not None:
        apex = target_pos + BCONE_OFFSET * v_axis
        rim, spokes = cone_wire(apex, -v_axis, BCONE_TAN, CONE_H, rim_theta, spoke_theta)
    else:
        apex = target_pos + np.array([0.0, 0.0, -GS_APEX_OFFSET])
        rim, spokes = cone_wire(apex, np.array([0.0, 0.0, 1.0]), GS_TAN, CONE_H, rim_theta, spoke_theta)
    ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=CONE, lw=1.4, alpha=alpha)
    for sp in spokes:
        ax.plot(sp[:, 0], sp[:, 1], sp[:, 2], color=CONE, lw=0.65, alpha=alpha * 0.55)


def make_cone_artists(ax):
    spokes = [ax.plot([], [], [], color=CONE, lw=0.9, alpha=0.40, zorder=3)[0]
              for _ in range(CONE_N)]
    rim, = ax.plot([], [], [], color=CONE, lw=1.6, alpha=0.70, zorder=3)
    base, = ax.plot([], [], [], color=CONE, lw=0.5, alpha=0.20, ls=":", zorder=3)
    return {"rim": rim, "base": base, "spokes": spokes}


def update_cone_artists(artists, mode: str, target_pos, v_axis=None, alpha=CONE_ALPHA):
    target_pos = np.asarray(target_pos, dtype=float)
    rim_theta = np.linspace(0, 2 * np.pi, 80)
    spoke_theta = np.linspace(0, 2 * np.pi, CONE_N, endpoint=False)
    if mode == "bcone" and v_axis is not None:
        axis = np.asarray(v_axis, dtype=float)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        apex = target_pos + BCONE_OFFSET * axis
        rim, spokes = cone_wire(apex, -axis, BCONE_TAN, CONE_H, rim_theta, spoke_theta)
        artists["rim"].set_data_3d(rim[:, 0], rim[:, 1], rim[:, 2])
        artists["base"].set_data_3d([], [], [])
        for line, sp in zip(artists["spokes"], spokes):
            line.set_data_3d(sp[:, 0], sp[:, 1], sp[:, 2])
    else:
        apex = target_pos + np.array([0.0, 0.0, -GS_APEX_OFFSET])
        rim, spokes = cone_wire(apex, np.array([0.0, 0.0, 1.0]), GS_TAN, CONE_H, rim_theta, spoke_theta)
        artists["rim"].set_data_3d(rim[:, 0], rim[:, 1], rim[:, 2])
        shadow_r = GS_TAN * max(0.0, -apex[2])
        artists["base"].set_data_3d(target_pos[0] + shadow_r * np.cos(rim_theta),
                                    target_pos[1] + shadow_r * np.sin(rim_theta),
                                    np.zeros_like(rim_theta))
        for line, sp in zip(artists["spokes"], spokes):
            line.set_data_3d(sp[:, 0], sp[:, 1], sp[:, 2])
    artists["rim"].set_alpha(alpha)
    artists["base"].set_alpha(alpha * 0.35)
    for line in artists["spokes"]:
        line.set_alpha(alpha * 0.75)


def cone_artist_list(artists):
    return [artists["rim"], artists["base"], *artists["spokes"]]


def set_limits(ax, dpos, tpos):
    pts = np.vstack([dpos, tpos])
    lo = np.nanmin(pts, axis=0)
    hi = np.nanmax(pts, axis=0)
    pad = np.maximum((hi - lo) * 0.12, np.array([0.7, 0.7, 0.4]))
    lo -= pad
    hi += pad
    lo[2] = min(0.0, lo[2])
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    return lo, hi


def apply_anim_3d_limits(ax):
    lo = np.array([ANIM_X_LIM[0], ANIM_Y_LIM[0], ANIM_Z_LIM[0]], dtype=float)
    hi = np.array([ANIM_X_LIM[1], ANIM_Y_LIM[1], ANIM_Z_LIM[1]], dtype=float)
    ax.set_xlim(*ANIM_X_LIM)
    ax.set_ylim(*ANIM_Y_LIM)
    ax.set_zlim(*ANIM_Z_LIM)
    return lo, hi


def finite_ylim(values, pad_frac=0.12, min_pad=0.1):
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return -1.0, 1.0
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if abs(hi - lo) < 1e-9:
        pad = max(abs(hi) * 0.1, min_pad)
    else:
        pad = max((hi - lo) * pad_frac, min_pad)
    return lo - pad, hi + pad


def maybe_deg(values):
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite) and np.nanmax(np.abs(finite)) <= 2.0 * np.pi + 0.25:
        return np.rad2deg(vals)
    return vals


def first_series(df, names, default=0.0):
    for name in names:
        if name in df.columns:
            return col(df, name, default), name
    return np.full(len(df), default, dtype=float), names[0]


def stack_optional(df, names, default=0.0):
    return np.stack([col(df, name, default) for name in names], axis=1)


def rh_node_indices(rh):
    if not len(rh):
        return np.array([], dtype=int)
    return np.flatnonzero(np.r_[True, np.diff(rh) != 0])


def solve_time_panel(data):
    solves = data["solves"]
    if "solve_id" in solves.columns:
        key, ms_name = "solve_id", "solve_ms"
    else:
        key, ms_name = "rh", "ms"
    if key not in solves.columns or ms_name not in solves.columns:
        ids = np.unique(data["rh"])
        return {"type": "bar", "title": "Solve time", "ylabel": "[ms]", "x": ids, "y": np.zeros_like(ids, dtype=float)}
    xs, ys = [], []
    for sid, grp in solves.groupby(key):
        xs.append(int(sid))
        ys.append(float(pd.to_numeric(grp[ms_name], errors="coerce").iloc[0]))
    return {"type": "bar", "title": "Solve time", "ylabel": "[ms]", "x": np.asarray(xs), "y": np.asarray(ys)}


def telemetry_panels(kind: str, data: dict, cap: float, limits: dict | None = None):
    df = data["exec"]
    time = data["time"]
    dpos = data["dpos"]
    tpos = data["tpos"]
    thrust, thrust_name = first_series(df, ("T", "fz_B"))
    thrust_label = "Collective thrust, $T$ [N]" if thrust_name == "T" else "Body force, $f_z^B$ [N]"

    if all(name in df.columns for name in ("Mx", "My", "Mz")):
        control = stack_optional(df, ("Mx", "My", "Mz"))
        control_title = "Body moments, $M_B/I_{xx}$ [rad s$^{-2}$]"
        control_labels = ("$M_x$", "$M_y$", "$M_z$")
    else:
        control = stack_optional(df, ("wx_cmd", "wy_cmd", "wz_cmd"))
        control_title = "Commanded body rates, $\\omega_{cmd}$ [rad s$^{-1}$]"
        control_labels = ("$\\omega_x^c$", "$\\omega_y^c$", "$\\omega_z^c$")

    if kind == "target":
        p_title = "Target-frame position, $p_B^N$ [m]"
        v_title = "Target-frame velocity, $v_B^N$ [m s$^{-1}$]"
        p_labels = ("$N_x$", "$N_y$", "$N_z$")
        v_labels = ("$v_{N_x}$", "$v_{N_y}$", "$v_{N_z}$")
    elif kind == "body":
        p_title = "Body-frame target position, $p_T^B$ [m]"
        v_title = "Body-frame relative velocity, $v_{rel}^B$ [m s$^{-1}$]"
        p_labels = ("$b_x$", "$b_y$", "$b_z$")
        v_labels = ("$v_{b_x}$", "$v_{b_y}$", "$v_{b_z}$")
    else:
        p_title = "Relative position, $p_{D}-p_{T}$ [m]"
        v_title = "Relative velocity, $v_{D}-v_{T}$ [m s$^{-1}$]"
        p_labels = ("$r_x$", "$r_y$", "$r_z$")
        v_labels = ("$\\dot r_x$", "$\\dot r_y$", "$\\dot r_z$")

    margin_title = "Backward-cone margin [m]" if data["mode"] == "bcone" else "Glideslope margin [m]"

    lim = limits or {}
    omega_hlines = []
    control_hlines = []
    if "omega_max" in lim:
        om = float(lim["omega_max"])
        omega_hlines = [(om, "#d62728"), (-om, "#d62728")]
        if "wx_cmd" in df.columns:
            control_hlines = [(om, "#d62728"), (-om, "#d62728")]
    quat = stack_optional(df, ("qw", "qx", "qy", "qz"))
    phase = data["phase"]
    alt_trigger = data.get("alt_trigger", np.zeros_like(phase))
    trigger_active = data["trigger_active"]
    stc_active = data["stc_active"]
    phase_switch_time = data.get("phase_switch_time")
    switch_times = [] if phase_switch_time is None else [phase_switch_time]
    stc_cap = float(lim.get("ctcs_beta", 1e-3)) if "ctcs_beta" in lim else 1e-3

    n_pts = len(alt_trigger)
    has_ctcs = bool(data.get("has_ctcs", False))
    state_mask = np.asarray(data.get("ctcs_altitude_state_active", stc_active), dtype=bool)
    thrust_mask = np.asarray(data.get("ctcs_thrust_active", stc_active), dtype=bool)

    def phase1_bound_overlays(key, label, color=INACTIVE):
        if key not in lim:
            return []
        v = float(lim[key])
        return [
            {"y": np.full(n_pts, v),  "color": color, "ls": "-", "alpha": 0.55, "label": f"+{label}"},
            {"y": np.full(n_pts, -v), "color": color, "ls": "-", "alpha": 0.55, "label": f"-{label}"},
        ]

    def triggered_bound_overlays(key, label, color=PHASE, mask=None):
        if key not in lim:
            return []
        v = float(lim[key])
        mask = alt_trigger > 1e-6 if mask is None else np.asarray(mask, dtype=bool)
        return [
            {"y": np.where(mask, v, np.nan),  "color": color, "ls": "--", "alpha": 0.88, "label": f"+{label}"},
            {"y": np.where(mask, -v, np.nan), "color": color, "ls": "--", "alpha": 0.88, "label": f"-{label}"},
        ]

    def upper_bound_overlay(key, label, mask=None, color=PHASE, ls="--", alpha=0.88):
        if key not in lim:
            return []
        v = float(lim[key])
        y = np.full(n_pts, v) if mask is None else np.where(np.asarray(mask, dtype=bool), v, np.nan)
        return [{"y": y, "color": color, "ls": ls, "alpha": alpha, "label": label}]

    def scalar_bound_overlay(value, label, mask=None, color=PHASE, ls="--", alpha=0.88):
        y = np.full(n_pts, float(value)) if mask is None else np.where(np.asarray(mask, dtype=bool), float(value), np.nan)
        return [{"y": y, "color": color, "ls": ls, "alpha": alpha, "label": label}]

    def degree_bound_overlay(key, label, mask=None, color=PHASE, ls="--", alpha=0.88):
        if key not in lim:
            return []
        y = np.full(n_pts, float(lim[key])) if mask is None else np.where(np.asarray(mask, dtype=bool), float(lim[key]), np.nan)
        return [{"y": y, "color": color, "ls": ls, "alpha": alpha, "label": label}]

    def thrust_bound_overlays(mask):
        overlays = []
        if "fmax" in lim:
            overlays.append({"y": np.full(n_pts, float(lim["fmax"])), "color": INACTIVE, "ls": "-", "alpha": 0.55, "label": "$f_{max}$"})
        if "fmin" in lim:
            overlays.append({"y": np.full(n_pts, float(lim["fmin"])), "color": INACTIVE, "ls": "-", "alpha": 0.55, "label": "$f_{min}$"})
        if "fmax_land" in lim:
            overlays.append({"y": np.where(mask, float(lim["fmax_land"]), np.nan), "color": PHASE, "ls": "--", "alpha": 0.88, "label": "$f_{max,land}$"})
        if "fmin_land" in lim:
            overlays.append({"y": np.where(mask, float(lim["fmin_land"]), np.nan), "color": PHASE, "ls": "--", "alpha": 0.88, "label": "$f_{min,land}$"})
        return overlays

    speed_bound = float(lim["v_phase0_max"]) if "v_phase0_max" in lim else None
    speed_overlays = (phase1_bound_overlays("v_phase0_max", "$v_{max}$") +
                      triggered_bound_overlays("v_land_stc_max", "$v^{stc}$"))

    mask = alt_trigger > 1e-6


    tilt_overlays = []
    if "tilt_phase0_deg" in lim:
        phase1_t = np.sin(0.5 * np.radians(float(lim["tilt_phase0_deg"])))
        tilt_overlays += [{"y": np.full(n_pts,  phase1_t), "color": INACTIVE, "ls": "-", "alpha": 0.55, "label": "$\\theta_{max}$"},
                          {"y": np.full(n_pts, -phase1_t), "color": INACTIVE, "ls": "-", "alpha": 0.55, "label": ""}]
    if "tilt_land_stc_deg" in lim:
        triggered_t = np.sin(0.5 * np.radians(float(lim["tilt_land_stc_deg"])))
        tilt_overlays += [{"y": np.where(mask,  triggered_t, np.nan), "color": PHASE, "ls": "--", "alpha": 0.88, "label": "$\\theta^{stc}$"},
                          {"y": np.where(mask, -triggered_t, np.nan), "color": PHASE, "ls": "--", "alpha": 0.88, "label": ""}]

    omega_overlays = (phase1_bound_overlays("omega_phase0_max", "$\\omega_{max}$") +
                      triggered_bound_overlays("omega_land_stc_max", "$\\omega^{stc}$"))

    thrust_overlays = []
    if "fmax" in lim:
        thrust_overlays.append({"y": np.full(n_pts, float(lim["fmax"])), "color": INACTIVE, "ls": "-", "alpha": 0.55, "label": "$f_{max}$"})
    if "fmax_land" in lim:
        thrust_overlays.append({"y": np.where(mask, float(lim["fmax_land"]), np.nan), "color": PHASE, "ls": "--", "alpha": 0.88, "label": "$f_{max,land}$"})
    if "fmin" in lim:
        thrust_overlays.append({"y": np.full(n_pts, float(lim["fmin"])), "color": INACTIVE, "ls": "-", "alpha": 0.55, "label": "$f_{min}$"})
    if "fmin_land" in lim:
        thrust_overlays.append({"y": np.where(mask, float(lim["fmin_land"]), np.nan), "color": PHASE, "ls": "--", "alpha": 0.88, "label": "$f_{min,land}$"})

    margin_y = data["margin"]
    if data["mode"] != "bcone" and "gs_cone_tan" in lim:
        rel_xy = np.linalg.norm(data["local"][:, :2], axis=1)
        rel_z = data["local"][:, 2] + float(lim.get("gs_apex_offset", GS_APEX_OFFSET))
        cone_tan = float(lim["gs_cone_tan"])
        margin_y = cone_tan * np.maximum(0.0, rel_z) - rel_xy
        margin_y = np.where(stc_active, margin_y, np.nan)
    elif data["mode"] != "bcone" and "gs_stc_tan" in lim:
        rel_xy = np.linalg.norm(data["local"][:, :2], axis=1)
        rel_z = data["local"][:, 2] + float(lim.get("gs_apex_offset", GS_APEX_OFFSET))
        stc_tan = float(lim["gs_stc_tan"])
        margin_y = np.maximum(0.0, rel_z) / max(stc_tan, 1e-9) - rel_xy
        margin_y = np.where(stc_active, margin_y, np.nan)

    if has_ctcs:
        ctcs_speed_overlays = []
        ctcs_speed_overlays += upper_bound_overlay("v_phase0_max", "$v_{max}$", None, INACTIVE, "-", 0.55)
        ctcs_speed_overlays += upper_bound_overlay("v_land_stc_max", "$v^{stc}$", state_mask)

        ctcs_tilt_overlays = []
        ctcs_tilt_overlays += degree_bound_overlay("tilt_phase0_deg", "$\\theta_{max}$", None, INACTIVE, "-", 0.55)
        ctcs_tilt_overlays += degree_bound_overlay("tilt_land_stc_deg", "$\\theta^{stc}$", state_mask)

        ctcs_omega_overlays = []
        ctcs_omega_overlays += upper_bound_overlay("omega_phase0_max", "$\\omega_{max}$", None, INACTIVE, "-", 0.55)
        ctcs_omega_overlays += upper_bound_overlay("omega_land_stc_max", "$\\omega^{stc}$", state_mask)
        state_violations = np.column_stack([
            data["stc_speed_violation"],
            data["stc_tilt_violation"],
            data["stc_omega_violation"],
            data["gs_violation"],
        ])
        state_violation_labels = ("speed", "tilt angle", "angular rate", "glideslope")
        state_violation_colors = (DRONE, PHASE, BZ, CONE)
        state_violations = np.where(state_mask[:, None], state_violations, np.nan)
        thrust_violations = np.column_stack([
            data["stc_thrust_lo_violation"],
            data["stc_thrust_hi_violation"],
        ])
        thrust_violations = np.where(thrust_mask[:, None], thrust_violations, np.nan)

        antecedent_y      = np.column_stack([alt_trigger, data["ctcs_a_altitude_state"], data["ctcs_a_thrust"]])
        antecedent_labels = ("altitude trigger", "$A_{state}$", "$A_{thrust}$")
        antecedent_colors = (CONE, PHASE, THRUST)
        return [
            {"title": "CT-cSTC antecedents", "ylabel": "",
             "y": antecedent_y,
             "labels": antecedent_labels,
             "colors": antecedent_colors, "active": state_mask, "switch_times": switch_times},
            {"title": "Speed $\\|v\\|_2$ [m s$^{-1}$]", "ylabel": "",
             "y": data["stc_rel_speed"], "color": DRONE,
             "overlays": ctcs_speed_overlays, "active": state_mask, "switch_times": switch_times},
            {"title": "Tilt angle $\\theta$ [deg]", "ylabel": "",
             "y": data["stc_tilt_deg"], "color": PHASE,
             "overlays": ctcs_tilt_overlays, "active": state_mask, "switch_times": switch_times},
            {"title": "Angular rate $\\|\\omega\\|_2$ [rad s$^{-1}$]", "ylabel": "",
             "y": data["stc_omega_norm"], "color": BZ,
             "overlays": ctcs_omega_overlays, "active": state_mask, "switch_times": switch_times},
            {"title": "Glideslope violation", "ylabel": "",
             "y": data["gs_violation"], "color": CONE, "zero": True,
             "active": state_mask, "switch_times": switch_times},
            {"title": "Thrust force $T$ [N]", "ylabel": "",
             "y": data["stc_fz_b"], "color": THRUST,
             "overlays": thrust_bound_overlays(thrust_mask), "active": thrust_mask, "switch_times": switch_times},
            {"title": "CT-cSTC accumulator/integrand", "ylabel": "",
             "y": np.column_stack([data["ctcs_y"], data["ctcs_integrand"]]),
             "labels": ("$y$", "$h_{ctcs}$"), "colors": (PHASE, VIOL),
             "active": state_mask, "switch_times": switch_times},
            {"title": "Altitude-triggered state constraint violations", "ylabel": "",
             "y": state_violations,
             "labels": state_violation_labels,
             "colors": state_violation_colors, "zero": True,
             "active": state_mask, "switch_times": switch_times},
            {"title": "Speed/tilt-triggered thrust force violations", "ylabel": "",
             "y": thrust_violations,
             "labels": ("$f_{min}$", "$f_{max}$"),
             "colors": (THRUST, VIOL), "zero": True,
             "active": thrust_mask, "switch_times": switch_times},
            {"title": p_title, "ylabel": "", "y": data["local"], "labels": p_labels},
            {"title": v_title, "ylabel": "", "y": data["local_v"], "labels": v_labels},
            {"title": "Altitude / distance [m]", "ylabel": "", "y": np.column_stack([dpos[:, 2], data["dist"]]), "labels": ("$z$", "$e_p$"), "colors": (BZ, TARGET), "cap": cap},
            {"title": thrust_label, "ylabel": "", "y": thrust, "color": TARGET, "overlays": thrust_bound_overlays(thrust_mask)},
            solve_time_panel(data),
        ]

    panels = [
        {"title": "Capture phase / CT-cSTC altitude trigger", "ylabel": "", "y": np.column_stack([phase, alt_trigger]),
         "labels": ("capture_phase", "alt_trigger"), "colors": (PHASE, CONE),
         "hlines": [(PHASE_VIS_THRESHOLD, INACTIVE), (PHASE_ON_THRESHOLD, PHASE)], "switch_times": switch_times},
        {"title": "Speed bounds [m s$^{-1}$]", "ylabel": "",
         "y": data["local_v"], "labels": v_labels,
         "overlays": speed_overlays, "active": stc_active, "switch_times": switch_times},
        {"title": "Tilt attitude components", "ylabel": "",
         "y": data["stc_tilt_qxy"], "labels": ("$q_x$", "$q_y$"), "colors": (BX, BY),
         "overlays": tilt_overlays, "active": stc_active, "switch_times": switch_times},
        {"title": "Angular-rate bounds [rad s$^{-1}$]", "ylabel": "",
         "y": stack_optional(df, ("wx", "wy", "wz")), "labels": ("$\\omega_x$", "$\\omega_y$", "$\\omega_z$"),
         "overlays": omega_overlays, "active": stc_active, "switch_times": switch_times},
        {"title": "Thrust force bounds [N]", "ylabel": "",
         "y": data["stc_fz_b"], "color": THRUST,
         "overlays": thrust_overlays, "active": stc_active, "switch_times": switch_times},
        {"title": "CT-cSTC increment / cap (Theta*density <= CTCS_BETA)", "ylabel": "",
         "y": data["stc_inc"], "color": PHASE,
         "hlines": [(stc_cap, PHASE, stc_active)], "active": stc_active, "switch_times": switch_times},
        {"title": margin_title, "ylabel": "",
         "y": margin_y, "color": CONE, "zero": True,
         "active": stc_active, "switch_times": switch_times},
        {"title": p_title, "ylabel": "", "y": data["local"], "labels": p_labels},
        {"title": v_title, "ylabel": "", "y": data["local_v"], "labels": v_labels},
        {"title": "Altitude / distance [m]", "ylabel": "", "y": np.column_stack([dpos[:, 2], data["dist"]]), "labels": ("$z$", "$e_p$"), "colors": (BZ, TARGET), "cap": cap},
        {"title": thrust_label, "ylabel": "", "y": thrust, "color": TARGET, "overlays": thrust_overlays},
        {"title": "Speed, $\\|v\\|_2$ [m s$^{-1}$]", "ylabel": "", "y": data["speed"], "color": DRONE},
        {"title": control_title, "ylabel": "", "y": control, "labels": control_labels, "hlines": control_hlines},
        {"title": "Angular velocity, $\\omega_B$ [rad s$^{-1}$]", "ylabel": "", "y": stack_optional(df, ("wx", "wy", "wz")), "labels": ("$\\omega_x$", "$\\omega_y$", "$\\omega_z$"), "hlines": omega_hlines},
        solve_time_panel(data),
    ]
    return panels


def plot_telemetry_panel(ax, panel, time, nodes=None, upto=None):
    panel_type = panel.get("type", "line")
    if panel_type == "bar":
        x = panel["x"]
        y = panel["y"]
        ax.bar(x, y, color=SOLVE, alpha=0.55, width=0.7)
        if len(y):
            ax.axhline(np.nanmean(y), color=SOLVE, lw=0.9, ls="--", alpha=0.55)
            ax.set_xlim(float(np.nanmin(x)) - 0.75, float(np.nanmax(x)) + 0.75)
        style_panel(ax, panel["title"], panel.get("ylabel", ""))
        ax.set_xlabel("RH iter", fontsize=8)
        return []

    y = np.asarray(panel["y"], dtype=float)
    if upto is not None:
        xs = time[:upto + 1]
        ys = y[:upto + 1]
    else:
        xs = time
        ys = y
    colors = panel.get("colors", (BX, BY, BZ))
    artists = []
    active = panel.get("active")
    if active is not None and len(time):
        active_arr = np.asarray(active, dtype=bool)
        if upto is not None:
            active_arr = active_arr[:upto + 1]
        if len(active_arr):
            xs_active = xs[:len(active_arr)]
            ax.fill_between(
                xs_active, 0.0, 1.0,
                where=active_arr,
                transform=ax.get_xaxis_transform(),
                color=CONE,
                alpha=0.08,
                linewidth=0.0,
                label="constraint ON" if y.ndim == 1 else None,
            )
    if y.ndim == 2:
        labels = panel.get("labels", tuple(f"${c}$" for c in "xyz"))
        for j in range(y.shape[1]):
            line, = ax.plot(xs, ys[:, j], color=colors[j % len(colors)], lw=1.15, label=labels[j])
            artists.append(line)
            if nodes is not None and upto is None:
                valid = nodes[nodes < len(time)]
                ax.scatter(time[valid], y[valid, j], s=9, color=colors[j % len(colors)], alpha=0.65, zorder=4)
        ax.legend(fontsize=6, framealpha=0.72, loc="best")
    else:
        line, = ax.plot(xs, ys, color=panel.get("color", DRONE), lw=1.25)
        artists.append(line)
        if nodes is not None and upto is None:
            valid = nodes[nodes < len(time)]
            ax.scatter(time[valid], y[valid], s=10, color="black", alpha=0.7, zorder=4)

    overlay_values = []
    for overlay in panel.get("overlays", []):
        oy = np.asarray(overlay.get("y", []), dtype=float)
        if upto is not None:
            oy = oy[:upto + 1]
        if len(oy) != len(xs):
            continue
        line, = ax.plot(
            xs, oy,
            color=overlay.get("color", PHASE),
            lw=1.0,
            ls=overlay.get("ls", "--"),
            alpha=overlay.get("alpha", 0.82),
            label=overlay.get("label"),
        )
        artists.append(line)
        overlay_values.append(oy)
    if panel.get("overlays"):
        ax.legend(fontsize=6, framealpha=0.72, loc="best")

    if panel.get("zero"):
        ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.55)
    if "cap" in panel:
        ax.axhline(panel["cap"], color=CONE, lw=0.8, ls="--", alpha=0.65)
    for hline in panel.get("hlines", []):
        hval, hcolor = hline[0], hline[1]
        hmask = hline[2] if len(hline) > 2 else None
        if hmask is None:
            ax.axhline(hval, color=hcolor, lw=0.85, ls="--", alpha=0.70)
            continue
        mask = np.asarray(hmask, dtype=bool)
        if upto is not None:
            mask = mask[:upto + 1]
        mask = mask[:len(xs)]
        if len(mask):
            ymask = np.full(len(xs), np.nan, dtype=float)
            ymask[:len(mask)] = np.where(mask, float(hval), np.nan)
            ax.plot(xs, ymask, color=hcolor, lw=0.95, ls="--", alpha=0.82)
    for st in panel.get("switch_times", []):
        if time[0] <= st <= time[-1]:
            ax.axvline(st, color=PHASE, lw=1.0, ls="--", alpha=0.75)
    ylim_values = [np.asarray(y)]
    ylim_values.extend(overlay_values)
    lo, hi = finite_ylim(np.concatenate([v.reshape(-1) for v in ylim_values if len(v.reshape(-1))]))
    if panel.get("zero"):
        lo, hi = min(lo, -0.1), max(hi, 0.1)
    if "cap" in panel:
        lo, hi = min(lo, 0.0), max(hi, panel["cap"] * 1.25)
    for hline in panel.get("hlines", []):
        hval = hline[0]
        lo, hi = min(lo, hval * 1.05 if hval < 0 else lo), max(hi, hval * 1.05 if hval > 0 else hi)
    ax.set_ylim(lo, hi)
    style_panel(ax, panel["title"], panel.get("ylabel", ""))
    ax.set_xlim(time[0], time[-1])
    return artists


def save_summary(path: str, kind: str, mode: str, data: dict, groups: dict, cap: float, limits: dict | None = None, only_3d: bool = False):
    dpos, tpos, time = data["dpos"], data["tpos"], data["time"]
    speed, rh = data["speed"], data["rh"]
    fig = plt.figure(figsize=(13.5, 10.5), facecolor=BG)
    gs = fig.add_gridspec(5, 3, height_ratios=[1.35, 1.35, 1.0, 1.0, 1.0], hspace=0.55, wspace=0.34)
    ax = fig.add_subplot(gs[:2, :2], projection="3d", computed_zorder=False)
    style_3d(ax, "Landing Trajectory Snapshot")
    lo, hi = set_limits(ax, dpos, tpos)

    lc = Line3DCollection(seg3d(dpos), cmap=plt.get_cmap(CMAP),
                          norm=mcolors.Normalize(0, max(float(np.nanmax(speed)), 1e-6)),
                          array=speed, lw=2.7, zorder=5)
    ax.add_collection3d(lc)
    fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02, shrink=0.78).set_label("Speed [m/s]", fontsize=8)
    if kind != "sh":
        ax.plot(tpos[:, 0], tpos[:, 1], tpos[:, 2], color=TARGET, lw=1.8, alpha=0.8, label="Target")
    ax.plot(dpos[:, 0], dpos[:, 1], np.zeros(len(dpos)), color="#777777", lw=0.9, alpha=0.65, label="XY projection")
    ax.plot(dpos[:, 0], np.full(len(dpos), hi[1]), dpos[:, 2], color="#777777", lw=0.9, alpha=0.55, label="XZ projection")
    ax.plot(np.full(len(dpos), lo[0]), dpos[:, 1], dpos[:, 2], color="#777777", lw=0.9, alpha=0.55, label="YZ projection")

    nodes = np.flatnonzero(np.r_[True, np.diff(rh) != 0])
    ax.scatter(dpos[nodes, 0], dpos[nodes, 1], dpos[nodes, 2], s=10, c="black", depthshade=False, label="RH nodes")
    switch_idx = np.flatnonzero(data["stc_active"])
    if len(switch_idx):
        si = int(switch_idx[0])
        ax.scatter([dpos[si, 0]], [dpos[si, 1]], [dpos[si, 2]],
                   s=58, c=PHASE, marker="*", depthshade=False, label="STC trigger")

    for idx in np.unique(np.linspace(0, len(dpos) - 1, min(9, len(dpos)), dtype=int)):
        draw_drone(ax, dpos[idx], data["quat"][idx], scale=0.65, alpha=0.55 if idx != len(dpos) - 1 else 0.95)
    draw_target_pad(ax, tpos[-1], data["yaw"][-1])

    last_rh = int(rh[-1]) if len(rh) else 0
    last_sid = solve_id_for_rh(last_rh, groups)
    if last_sid in groups:
        g = groups[last_sid]
        if g["drone"] is not None:
            ax.plot(g["drone"][:, 0], g["drone"][:, 1], g["drone"][:, 2], color=PLAN_D, lw=1.2, ls="--", label="Last RH plan")
            ax.scatter(g["drone"][:, 0], g["drone"][:, 1], g["drone"][:, 2], s=14, c="black", depthshade=False)
        if g["target"] is not None:
            ax.plot(g["target"][:, 0], g["target"][:, 1], g["target"][:, 2], color=PLAN_T, lw=1.1, ls="--")
        if g["pred"] is not None:
            ax.plot(g["pred"][:, 0], g["pred"][:, 1], g["pred"][:, 2], color=PRED_T, lw=1.1, ls=":")

    v_axis = data["v_hat"][-1] if data["v_hat"] is not None else None
    draw_constraint(ax, mode, tpos[-1], v_axis)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    try:
        ax.set_box_aspect((1.4, 1.2, 0.8))
    except Exception:
        pass
    ax.view_init(elev=28, azim=-45)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    ax_info = fig.add_subplot(gs[:2, 2])
    ax_info.axis("off")
    final_dist = data["dist"][-1] if len(data["dist"]) else np.nan
    min_margin = np.nanmin(data["margin"]) if len(data["margin"]) else np.nan
    max_speed = np.nanmax(speed) if len(speed) else np.nan
    phase_switch_time = data.get("phase_switch_time")
    phase_text = f"{phase_switch_time:.3f} s" if phase_switch_time is not None else "never"
    final_phase = data["phase"][-1] if len(data["phase"]) else np.nan
    ax_info.text(
        0.02, 0.95,
        "Summary\n"
        f"mode: {mode}\n"
        f"steps: {len(time)}\n"
        f"RH solves: {len(np.unique(rh))}\n"
        f"final dist: {final_dist:.3f} m\n"
        f"max speed: {max_speed:.3f} m/s\n"
        f"min margin: {min_margin:.3f} m\n"
        f"STC trigger: {phase_text}\n"
        f"final phase: {final_phase:.3f}",
        va="top", ha="left", fontsize=11, color=TEXT,
        bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.9),
    )

    local = data["local"]
    local_v = data["local_v"]
    dist = data["dist"]
    margin = data["margin"]
    tilt = np.array([np.degrees(np.arccos(np.clip(quat_to_rotmat(q)[2, 2], -1.0, 1.0))) for q in data["quat"]])
    rates = np.linalg.norm(np.gradient([quat_to_rotmat(q)[:, 2] for q in data["quat"]], time, axis=0), axis=1) if len(time) > 2 else np.zeros(len(time))

    if not only_3d:
        axes = [fig.add_subplot(gs[r, c]) for r in range(2, 5) for c in range(3)]
        panels = telemetry_panels(kind, data, cap, limits)
        nodes = rh_node_indices(rh)
        for axp, panel in zip(axes, panels[:9]):
            plot_telemetry_panel(axp, panel, time, nodes=nodes)
    fig.suptitle(f"{kind.upper()} landing summary", fontsize=14, fontweight="bold", color=TEXT)
    fig.savefig(path, dpi=180, facecolor=BG)
    plt.close(fig)


def animate(path: str | None, kind: str, mode: str, data: dict, groups: dict, fps: int, speed_mult: float, cap: float, limits: dict | None = None, variant: str = "", only_3d: bool = False):
    dpos, tpos, time, speed = data["dpos"], data["tpos"], data["time"], data["speed"]
    local = data["local"]
    rh = data["rh"]
    v_max = max(float(np.nanpercentile(speed, 98)), 0.1)
    cmap = mpl.colormaps[CMAP]
    norm = mcolors.Normalize(0, v_max)

    show_local = variant != "rh_landing" and not only_3d
    if kind == "target":
        local_title = "Target frame"
    elif kind == "body":
        local_title = "Body frame"
    else:
        local_title = "Shifted frame"

    fig = plt.figure(figsize=ANIM_FIGSIZE if not only_3d else (18, 12), facecolor=BG)
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=(1.0, 0.001) if only_3d else ANIM_HEIGHT_RATIOS,
        hspace=0.20,
        left=0.035, right=0.965, top=0.94, bottom=0.055,
    )
    if show_local:
        gs3 = outer[0].subgridspec(
            1, 3,
            width_ratios=[ANIM_3D_WIDTH, ANIM_GRADIENT_WIDTH, ANIM_3D_WIDTH],
            wspace=0.08,
        )
        axw = fig.add_subplot(gs3[0, 0], projection="3d")
        cax = fig.add_subplot(gs3[0, 1])
        axl = fig.add_subplot(gs3[0, 2], projection="3d")
        style_3d(axw, "World frame")
        style_3d(axl, local_title)
    else:
        gs3 = outer[0].subgridspec(
            1, 2,
            width_ratios=[ANIM_3D_WIDTH, ANIM_GRADIENT_WIDTH],
            wspace=0.08,
        )
        axw = fig.add_subplot(gs3[0, 0], projection="3d")
        cax = fig.add_subplot(gs3[0, 1])
        axl = None
        style_3d(axw, "World frame")

    if ANIM_USE_FIXED_3D_LIMITS:
        lo, hi = apply_anim_3d_limits(axw)
    else:
        lo, hi = set_limits(axw, dpos, tpos)
    try:
        axw.set_box_aspect((1.15, 1.15, 0.85))
    except Exception:
        pass
    axw.set_xlabel("X [m]")
    axw.set_ylabel("Y [m]")
    axw.set_zlabel("Z [m]")

    if show_local:
        if ANIM_USE_FIXED_3D_LIMITS:
            apply_anim_3d_limits(axl)
        else:
            lmax = max(float(np.nanmax(np.abs(local))), 1.5) + 0.5
            axl.set_xlim(-lmax, lmax)
            axl.set_ylim(-lmax, lmax)
            axl.set_zlim(-lmax, lmax)
        axl.set_xlabel("x [m]")
        axl.set_ylabel("y [m]")
        axl.set_zlabel("z [m]")
        try:
            axl.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

    world_lc = Line3DCollection([], cmap=cmap, norm=norm, lw=2.4)
    axw.add_collection3d(world_lc)
    if show_local:
        local_lc = Line3DCollection([], cmap=cmap, norm=norm, lw=2.4)
        axl.add_collection3d(local_lc)
    else:
        local_lc = None
    cbar = fig.colorbar(world_lc, cax=cax)
    cbar.set_label("Speed [m/s]", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    target_line, = axw.plot([], [], [], color=TARGET, lw=1.7)
    proj_xy, = axw.plot([], [], [], color="#777777", lw=0.9, alpha=0.65)
    proj_xz, = axw.plot([], [], [], color="#777777", lw=0.9, alpha=0.55)
    proj_yz, = axw.plot([], [], [], color="#777777", lw=0.9, alpha=0.55)
    node_world = axw.scatter([], [], [], s=10, c="black", depthshade=False, zorder=20)
    plan_w, = axw.plot([], [], [], color=PLAN_D, lw=1.3, ls="--", alpha=0.65)
    plan_t, = axw.plot([], [], [], color=PLAN_T, lw=1.2, ls="--", alpha=0.65)
    plan_pred, = axw.plot([], [], [], color=PRED_T, lw=1.2, ls=":", alpha=0.75)
    target_art_w = make_target_pad_artists(axw)
    drone_art_w = make_drone_artists(axw)
    world_cone = make_cone_artists(axw)
    axis_len = max(0.35, min(1.2, 0.22 * max(hi - lo)))
    body_axes = [axw.plot([], [], [], color=c, lw=2.0, zorder=14)[0] for c in (BX, BY, BZ)]

    if show_local:
        plan_l, = axl.plot([], [], [], color=PLAN_D, lw=1.3, ls="--", alpha=0.65)
        drone_art_l = make_drone_artists(axl)
        target_art_l = make_target_pad_artists(axl)
        local_cone = make_cone_artists(axl)
        local_axes = [axl.plot([], [], [], color=c, lw=2.0, zorder=14)[0] for c in (BX, BY, BZ)]

    title = fig.suptitle("", x=0.5, ha="center", color=TEXT, fontsize=13, fontweight="bold")

    nodes = rh_node_indices(rh)
    cursors = []
    if not only_3d:
        panels = telemetry_panels(kind, data, cap, limits)
        panel_grid = [panel for panel in panels if panel["title"] != "Altitude / distance [m]"][:9]
        info = outer[1].subgridspec(3, 3, hspace=0.62, wspace=0.30)
        for idx, panel in enumerate(panel_grid):
            axp = fig.add_subplot(info[idx // 3, idx % 3])
            plot_telemetry_panel(axp, panel, time, nodes=nodes)
            if panel.get("type") == "bar":
                cursor = axp.axvline(rh[0] if len(rh) else 0, color=TEXT, lw=1.0, alpha=0.62)
            else:
                cursor = axp.axvline(time[0], color=TEXT, lw=0.8, alpha=0.45)
            cursors.append((cursor, panel))

    def update(frame):
        i = min(frame, len(time) - 1)
        if i > 0:
            world_lc.set_segments(seg3d(dpos[:i + 1]))
            world_lc.set_array(speed[:i + 1])
            if show_local:
                local_lc.set_segments(seg3d(local[:i + 1]))
                local_lc.set_array(speed[:i + 1])
        proj_xy.set_data_3d(dpos[:i + 1, 0], dpos[:i + 1, 1], np.zeros(i + 1))
        proj_xz.set_data_3d(dpos[:i + 1, 0], np.full(i + 1, hi[1]), dpos[:i + 1, 2])
        proj_yz.set_data_3d(np.full(i + 1, lo[0]), dpos[:i + 1, 1], dpos[:i + 1, 2])
        visible_nodes = nodes[nodes <= i]
        if len(visible_nodes):
            node_world._offsets3d = (dpos[visible_nodes, 0], dpos[visible_nodes, 1], dpos[visible_nodes, 2])
        else:
            node_world._offsets3d = ([], [], [])
        if kind != "sh":
            target_line.set_data_3d(tpos[:i + 1, 0], tpos[:i + 1, 1], tpos[:i + 1, 2])
        update_target_pad(target_art_w, tpos[i], data["yaw"][i])
        update_drone(drone_art_w, dpos[i], data["quat"][i], scale=0.85)
        rmat = quat_to_rotmat(data["quat"][i])
        for axis_id, line in enumerate(body_axes):
            end = dpos[i] + axis_len * rmat[:, axis_id]
            line.set_data_3d([dpos[i, 0], end[0]], [dpos[i, 1], end[1]], [dpos[i, 2], end[2]])
        v_axis = data["v_hat"][i] if data["v_hat"] is not None else None
        constraint_on = bool(data["stc_active"][i])
        cone_alpha = CONE_ALPHA if constraint_on else 0.0
        update_cone_artists(world_cone, mode, tpos[i], v_axis, alpha=cone_alpha)

        local_artist_list = []
        if show_local:
            if kind == "body":
                update_drone(drone_art_l, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), scale=0.85)
                update_target_pad(target_art_l, local[i], data["yaw"][i])
                local_axis_origin = np.zeros(3)
            else:
                update_target_pad(target_art_l, np.zeros(3), data["yaw"][i])
                update_drone(drone_art_l, local[i], data["quat"][i], scale=0.85)
                local_axis_origin = local[i]
            for axis_id, line in enumerate(local_axes):
                end = local_axis_origin + axis_len * np.eye(3)[:, axis_id]
                line.set_data_3d([local_axis_origin[0], end[0]], [local_axis_origin[1], end[1]], [local_axis_origin[2], end[2]])
            local_axis = v_axis if data["v_hat"] is not None and kind == "rh" else np.array([1.0, 0.0, 0.0])
            update_cone_artists(local_cone, mode, np.zeros(3), local_axis, alpha=cone_alpha)
            rid = solve_id_for_rh(int(rh[i]), groups)
            if rid in groups:
                g = groups[rid]
                plan_l.set_data_3d(g["local"][:, 0], g["local"][:, 1], g["local"][:, 2])
            local_artist_list = (
                [local_lc, plan_l, *local_axes]
                + target_pad_artist_list(target_art_l)
                + drone_artist_list(drone_art_l)
                + cone_artist_list(local_cone)
            )

        rid = solve_id_for_rh(int(rh[i]), groups)
        if rid in groups:
            g = groups[rid]
            plan_w.set_data_3d(g["drone"][:, 0], g["drone"][:, 1], g["drone"][:, 2]) if g["drone"] is not None else plan_w.set_data_3d([], [], [])
            plan_t.set_data_3d(g["target"][:, 0], g["target"][:, 1], g["target"][:, 2]) if g["target"] is not None else plan_t.set_data_3d([], [], [])
            plan_pred.set_data_3d(g["pred"][:, 0], g["pred"][:, 1], g["pred"][:, 2]) if g["pred"] is not None else plan_pred.set_data_3d([], [], [])
        for cursor, panel in cursors:
            if panel.get("type") == "bar":
                cursor.set_xdata([rh[i], rh[i]])
            else:
                cursor.set_xdata([time[i], time[i]])
        constraint_status = "ON" if constraint_on else "OFF"
        title.set_text(
            f"{kind.upper()} landing  t={time[i]:.2f}s  dist={data['dist'][i]:.3f}m  "
            f"capture={data['phase'][i]:.3f}  CT-cSTC={constraint_status}"
        )
        return (
            [world_lc, target_line, proj_xy, proj_xz, proj_yz, node_world,
             plan_w, plan_t, plan_pred, title, *body_axes]
            + target_pad_artist_list(target_art_w)
            + drone_artist_list(drone_art_w)
            + cone_artist_list(world_cone)
            + local_artist_list
            + [cursor for cursor, _panel in cursors]
        )

    dt = np.diff(time)
    dt_pos = dt[dt > 1e-9]
    dt_med = float(np.median(dt_pos)) if len(dt_pos) else (1.0 / fps)
    writer_fps = max((1.0 / dt_med) * max(speed_mult, 1e-6), 1.0)
    interval = 1000.0 * dt_med / max(speed_mult, 1e-6)
    ani = animation.FuncAnimation(fig, update, frames=len(time), interval=interval, blit=False)
    if path is None:
        plt.show()
    elif path.lower().endswith(".gif"):
        ani.save(path, writer=animation.PillowWriter(fps=int(round(writer_fps))), dpi=ANIM_SAVE_DPI, savefig_kwargs={"facecolor": BG})
    else:
        ani.save(path, writer=animation.FFMpegWriter(fps=writer_fps, bitrate=2200), dpi=ANIM_SAVE_DPI, savefig_kwargs={"facecolor": BG})
    return ani


def main():
    global GS_DEG, GS_TAN, GS_APEX_OFFSET, BCONE_HALF_ANGLE_DEG, BCONE_TAN, BCONE_OFFSET

    parser = argparse.ArgumentParser(description="Unified one-file quad landing animator")
    parser.add_argument("--dir", required=True, help="Results directory")
    parser.add_argument("--out", default=None, help=".png/.pdf/.svg snapshot, .mp4/.gif animation, or omit for interactive")
    parser.add_argument("--kind", choices=["auto", "rh", "sh", "single-horizon", "tf", "bf", "target", "body", "target-frame", "body-frame", "rh-shifted"],
                        default="auto")
    parser.add_argument("--mode", choices=["gs", "bcone", "triggered_landing"], default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cap", type=float, default=0.15)
    parser.add_argument("--only-3d", action="store_true", help="Show only the 3D trajectory plot, no telemetry panels")
    args = parser.parse_args()

    base, kind, exec_df, solves_df = load_result(args.dir, args.kind)
    constraints = load_constraints(base)
    if kind == "sh":
        inferred_constraints = infer_single_horizon_stc_constraints()
        inferred_constraints.update(constraints)
        constraints = inferred_constraints
    if constraints:
        if "gs_deg" in constraints:
            GS_DEG = float(constraints["gs_deg"])
            GS_TAN = np.tan(np.radians(GS_DEG))
        if "gs_apex_offset" in constraints:
            GS_APEX_OFFSET = float(constraints["gs_apex_offset"])
        if "bcone_half_angle_deg" in constraints:
            BCONE_HALF_ANGLE_DEG = float(constraints["bcone_half_angle_deg"])
            BCONE_TAN = np.tan(np.radians(BCONE_HALF_ANGLE_DEG))
        if "bcone_offset" in constraints:
            BCONE_OFFSET = float(constraints["bcone_offset"])
    mode = args.mode or str(constraints.get("mode", "bcone" if "bcone" in str(base).lower() else "gs"))
    variant = str(constraints.get("variant", ""))
    limits = {}
    for k, v in constraints.items():
        try:
            limits[k] = float(v)
        except (ValueError, TypeError):
            pass
    if mode != "bcone" and "gs_cone_tan" in limits:
        GS_TAN = float(limits["gs_cone_tan"])
    elif mode != "bcone" and "gs_stc_tan" in limits:
        # cone_wire expects radius = tan_cone * height. The STC glideslope
        # constraint is tan(theta)*rxy <= height, so the drawable cone uses
        # tan_cone = 1/tan(theta).
        GS_TAN = 1.0 / max(float(limits["gs_stc_tan"]), 1e-9)
    print(f"kind={kind}  mode={mode}  variant={variant}  dir={base}")
    print(f"steps={len(exec_df)}  solves={solves_df.iloc[:, 0].nunique() if len(solves_df.columns) else 0}")
    data = build_data(kind, exec_df, solves_df, mode)
    groups = solve_groups(kind, solves_df)

    only_3d = args.only_3d
    if args.out and args.out.lower().endswith((".png", ".pdf", ".svg")):
        save_summary(args.out, kind, mode, data, groups, args.cap, limits, only_3d=only_3d)
    else:
        animate(args.out, kind, mode, data, groups, args.fps, args.speed, args.cap, limits, variant, only_3d=only_3d)


if __name__ == "__main__":
    main()
