"""Paper figure for state-triggered-constraint (STC) landing results.

Style follows `benchmark/papers/landing/STC/CT-cSTC/CT-cSTC.ipynb`:
  * 3D panel: speed-colored Line3DCollection (rainbow), gray back-wall
    projections, dashed altitude-trigger lines on the back walls, a single
    glideslope cone drawn as a dotted purple wireframe (notebook's
    `plot_cone`), and a few small drone glyphs sampled along the horizon.
  * State panels: blue state line, black node scatter, green dashed upper
    bound, purple dashed lower bound, red dashed trigger vline. Pre-switch
    bound runs from t=0 to t_trig, post-switch tightened bound runs from
    t_trig to t_end on the same panel.

The telemetry panels use target-relative states because those are the quantities
constrained by the STCs. The moving-target 3-D panel is plotted in world
coordinates, with the final glideslope cone attached to the final target pose.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# The benchmark's plotting module, renamed here because logs/animate.py is
# the (different) legacy landing animator for all_solves.csv.
import animate_benchmark as anim

mpl.rc("font", family="serif", serif=["Computer Modern Roman", "DejaVu Serif"])
mpl.rcParams.update({
    "font.size": 11,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
})

# ── Notebook color/size constants (CT-cSTC.ipynb) ────────────────────────────
C_STATE      = "#1f77b4"
C_NODE       = "#111111"
C_UPPER      = "#2ca02c"
C_LOWER      = "#6f42c1"
C_TRIG_ALT   = "#d62728"
C_TRIG_THR   = "#ff7f0e"
C_TRIG_STAGE = "#9467bd"
C_PROJ       = "#7a7a7a"
C_CONE       = "#6f42c1"
SCATTER_SC   = 9.0
X_FS         = 20
Y_FS         = 17
LEG_FS       = 18

# ── Constraint bounds (problem_examples/landing/quad_cf_landing_single_horizon_stc.cpp)
BOUNDS = dict(
    v_pre=3.5, v_post=0.8,
    omega_pre=2.5, omega_post=0.6,
    fmin=0.08, fmax=0.60,
    fmin_post=0.21, fmax_post=0.40,
    gs_pre_deg=35.0, gs_post_deg=15.0,
    gs_apex_offset=0.1,
    alt_trig=0.8,
    z_stage=1.8,
    los_alt_trig=1.8,
    los_trigger_scale=0.25,
    los_trigger_floor=1.0,
    los_cone_deg=30.0,
    los_cone_tan=float(np.tan(np.radians(30.0))),
    stage_radius=1.1,
    stage_trigger_scale=0.25,
    stage_trigger_floor=1.0,
    cap_radius=0.45,
    tilt_pre_deg=35.0,
    tilt_post_deg=5.0,
    thrust_speed_trig=1.0,
    thrust_tilt_trig_deg=15.0,
)


def resolve_result_dir(path: str) -> Path:
    base = Path(path)
    if base.exists():
        return base.resolve()
    repo = Path(__file__).resolve().parents[1]
    candidate = repo / base
    if candidate.exists():
        return candidate.resolve()
    return base.resolve()


def detect_dataset(base: Path, csv_name: str | None, kind: str) -> tuple[str, Path]:
    if kind != "auto":
        if csv_name is None:
            csv_name = {
                "single": "single_horizon_stc.csv",
                "single_horizon_stc": "single_horizon_stc.csv",
                "stateswitch_stc": "moving_executed.csv",
            }[kind]
        normalized = "single_horizon_stc" if kind == "single" else kind
        return normalized, base / csv_name

    if csv_name is not None:
        csv_path = base / csv_name
        header = pd.read_csv(csv_path, nrows=0).columns
        if {"tx", "ty", "tz"}.issubset(header):
            return "stateswitch_stc", csv_path
        return "single_horizon_stc", csv_path

    candidates = (
        ("single_horizon_stc", "single_horizon_stc.csv"),
        ("stateswitch_stc", "moving_executed.csv"),
        ("stateswitch_stc", "executed.csv"),
    )
    for detected_kind, name in candidates:
        path = base / name
        if path.exists():
            return detected_kind, path
    raise FileNotFoundError(
        f"No supported STC CSV found under {base}. "
        "Expected single_horizon_stc.csv or moving_executed.csv."
    )


def _parse_cpp_scalar(text: str, name: str) -> float | None:
    pattern = rf"\b{name}\b\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def load_current_bounds(kind: str, base: Path) -> dict:
    """Use result/source constants instead of stale plot defaults."""
    bounds = dict(BOUNDS)

    constraints_path = base / "constraints.csv"
    if constraints_path.exists():
        constraints = pd.read_csv(constraints_path)
        values = {}
        for _, row in constraints.iterrows():
            try:
                values[str(row["param"])] = float(row["value"])
            except (TypeError, ValueError):
                values[str(row["param"])] = row["value"]
        csv_mapping = {
            "v_phase0_max": "v_pre",
            "v_land_stc_max": "v_post",
            "omega_phase0_max": "omega_pre",
            "omega_land_stc_max": "omega_post",
            "fmin": "fmin",
            "fmax": "fmax",
            "fmin_land": "fmin_post",
            "fmax_land": "fmax_post",
            "gs_apex_offset": "gs_apex_offset",
            "alt_trig": "alt_trig",
            "z_stage": "z_stage",
            "los_alt_trig": "los_alt_trig",
            "los_trigger_scale": "los_trigger_scale",
            "los_trigger_floor": "los_trigger_floor",
            "los_cone_deg": "los_cone_deg",
            "los_cone_tan": "los_cone_tan",
            "stage_radius": "stage_radius",
            "stage_trigger_scale": "stage_trigger_scale",
            "stage_trigger_floor": "stage_trigger_floor",
            "cap_radius": "cap_radius",
            "tilt_phase0_deg": "tilt_pre_deg",
            "tilt_land_stc_deg": "tilt_post_deg",
        }
        for key, bound_name in csv_mapping.items():
            if key in values and isinstance(values[key], float):
                bounds[bound_name] = values[key]
        if "gs_cone_tan" in values and isinstance(values["gs_cone_tan"], float):
            bounds["gs_post_tan"] = values["gs_cone_tan"]
            bounds["gs_post_deg"] = float(np.degrees(np.arctan(bounds["gs_post_tan"])))
        elif "gs_stc_tan" in values and isinstance(values["gs_stc_tan"], float):
            # Moving-target STC source uses tan(theta) * r_xy <= z+z0, while
            # the drawable cone uses r_xy <= tan(alpha) * (z+z0).
            bounds["gs_constraint_tan"] = values["gs_stc_tan"]
            bounds["gs_constraint_deg"] = float(np.degrees(np.arctan(values["gs_stc_tan"])))
            bounds["gs_post_tan"] = 1.0 / max(values["gs_stc_tan"], 1e-12)
            bounds["gs_post_deg"] = float(np.degrees(np.arctan(bounds["gs_post_tan"])))
        elif "gs_deg" in values and isinstance(values["gs_deg"], float):
            bounds["gs_post_deg"] = values["gs_deg"]
            bounds["gs_post_tan"] = float(np.tan(np.radians(values["gs_deg"])))
        else:
            bounds["gs_post_tan"] = float(np.tan(np.radians(bounds["gs_post_deg"])))
        return bounds

    source_name = (
        "quad_cf_tracking_rh_landing_stateswitch_stc.cpp"
        if kind == "stateswitch_stc"
        else "quad_cf_landing_single_horizon_stc.cpp"
    )
    source = Path(__file__).resolve().parents[1] / "problem_examples/landing" / source_name
    try:
        text = source.read_text()
    except OSError:
        bounds["gs_post_tan"] = float(np.tan(np.radians(bounds["gs_post_deg"])))
        return bounds

    mapping = {
        "SPD_PHASE0_MAX": "v_pre",
        "SPD_STC_MAX": "v_post",
        "OMEGA_PHASE0_MAX": "omega_pre",
        "OMEGA_STC_MAX": "omega_post",
        "FMIN": "fmin",
        "FMAX": "fmax",
        "T_MIN_AFT": "fmin_post",
        "T_MAX_AFT": "fmax_post",
        "GS_CONE_HALF_ANGLE_DEG": "gs_post_deg",
        "GS_APEX_OFFSET": "gs_apex_offset",
        "ALT_TRIG": "alt_trig",
        "Z_STAGE": "z_stage",
        "LOS_ALT_TRIG": "los_alt_trig",
        "LOS_TRIGGER_SCALE": "los_trigger_scale",
        "LOS_TRIGGER_FLOOR": "los_trigger_floor",
        "LOS_CONE_HALF_ANGLE_DEG": "los_cone_deg",
        "LOS_CONE_TAN": "los_cone_tan",
        "STAGE_RADIUS": "stage_radius",
        "STAGE_TRIGGER_SCALE": "stage_trigger_scale",
        "STAGE_TRIGGER_FLOOR": "stage_trigger_floor",
        "CAP_RADIUS": "cap_radius",
        "CAP_COST_RADIUS": "cap_radius",
        "THETA_PHASE0_DEG": "tilt_pre_deg",
        "THETA_STC_DEG": "tilt_post_deg",
        "V_THRUST_TRIG": "thrust_speed_trig",
        "THETA_THRUST_TRIG_DEG": "thrust_tilt_trig_deg",
    }
    for cpp_name, bound_name in mapping.items():
        val = _parse_cpp_scalar(text, cpp_name)
        if val is not None:
            bounds[bound_name] = val
    if _parse_cpp_scalar(text, "GS_CONE_HALF_ANGLE_DEG") is None:
        old_horizontal = _parse_cpp_scalar(text, "GS_STC_DEG")
        if old_horizontal is not None:
            bounds["gs_constraint_deg"] = old_horizontal
            bounds["gs_constraint_tan"] = float(np.tan(np.radians(old_horizontal)))
            bounds["gs_post_tan"] = 1.0 / max(bounds["gs_constraint_tan"], 1e-12)
            bounds["gs_post_deg"] = float(np.degrees(np.arctan(bounds["gs_post_tan"])))
    else:
        bounds["gs_post_tan"] = float(np.tan(np.radians(bounds["gs_post_deg"])))
    return bounds


def read_and_normalize_dataset(base: Path, csv_path: Path, kind: str) -> tuple[pd.DataFrame, dict]:
    try:
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(csv_path)
    sort_cols = [c for c in ("node", "rh", "step", "t", "t_acc") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    if "t_acc" in df.columns:
        t = df["t_acc"].to_numpy(dtype=float)
    elif "t" in df.columns:
        t = df["t"].to_numpy(dtype=float)
        if len(t):
            t = t - t[0]
    else:
        t = np.arange(len(df), dtype=float)
    if not np.all(np.isfinite(t)) or np.all(t == 0):
        t = np.arange(len(df), dtype=float)

    if {"rx", "ry", "rz"}.issubset(df.columns):
        rxyz = df[["rx", "ry", "rz"]].to_numpy(dtype=float)
    elif {"x", "y", "z", "tx", "ty", "tz"}.issubset(df.columns):
        rxyz = (df[["x", "y", "z"]].to_numpy(dtype=float)
                - df[["tx", "ty", "tz"]].to_numpy(dtype=float))
    else:
        raise KeyError(f"{csv_path} has neither relative columns rx/ry/rz nor absolute x/y/z + tx/ty/tz")

    if {"rvx", "rvy", "rvz"}.issubset(df.columns):
        rvxyz = df[["rvx", "rvy", "rvz"]].to_numpy(dtype=float)
    elif {"vx", "vy", "vz", "tvx", "tvy", "tvz"}.issubset(df.columns):
        rvxyz = (df[["vx", "vy", "vz"]].to_numpy(dtype=float)
                 - df[["tvx", "tvy", "tvz"]].to_numpy(dtype=float))
    else:
        rvxyz = np.gradient(rxyz, axis=0)

    data = {
        "kind": kind,
        "csv_path": csv_path,
        "t": t,
        "rxyz": rxyz,
        "rvxyz": rvxyz,
        "world_xyz": df[["x", "y", "z"]].to_numpy(dtype=float)
                     if {"x", "y", "z"}.issubset(df.columns) else rxyz,
        "target_xyz": df[["tx", "ty", "tz"]].to_numpy(dtype=float)
                      if {"tx", "ty", "tz"}.issubset(df.columns) else np.zeros_like(rxyz),
        "quat": df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float),
        "omega": df[["wx", "wy", "wz"]].to_numpy(dtype=float),
        "fz": df["fz_B"].to_numpy(dtype=float),
        "df": df,
    }
    return df, data


# ── Notebook's plot_cone, lifted verbatim (minus the params plumbing). ───────
def plot_cone_wireframe(ax, center, direction, angle_deg, height,
                        cc=C_CONE, lw=0.5, label=None, zorder=15,
                        num_points=40):
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1e-12)

    theta = np.linspace(0.0, 2 * np.pi, num_points)
    r = np.linspace(0.0, height * np.tan(np.radians(angle_deg)), num_points)
    T, R = np.meshgrid(theta, r)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z = np.broadcast_to(np.linspace(0.0, height, num_points).reshape(-1, 1), X.shape)
    pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    rot = np.eye(3)
    if not np.allclose(direction, [0.0, 0.0, 1.0]):
        v = np.cross([0.0, 0.0, 1.0], direction)
        c = float(np.dot([0.0, 0.0, 1.0], direction))
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * ((1 - c) / max(np.linalg.norm(v) ** 2, 1e-12))
    rp = pts @ rot.T
    Xr = rp[:, 0].reshape(num_points, num_points) + center[0]
    Yr = rp[:, 1].reshape(num_points, num_points) + center[1]
    Zr = rp[:, 2].reshape(num_points, num_points) + center[2]

    ax.plot_wireframe(Xr, Yr, Zr, color=cc, linestyle="dotted",
                      lw=lw, zorder=zorder,
                      label=label, rstride=2, cstride=2)


def quat_to_rotmat(q):
    w, x, y, z = q
    n = max(np.linalg.norm(q), 1e-12)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def draw_quad_glyph(ax, center, quat, arm_len, zorder=18):
    """Compact quadrotor glyph: two crossed arms with small motor rings."""
    R = quat_to_rotmat(quat)
    c = np.asarray(center, dtype=float)
    for sign_x, sign_y in [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]:
        tip = c + R @ (arm_len * np.array([sign_x, sign_y, 0]) / np.sqrt(2))
        ax.plot([c[0], tip[0]], [c[1], tip[1]], [c[2], tip[2]],
                color="black", lw=1.6, zorder=zorder, solid_capstyle="round")
        # tiny motor disc
        th = np.linspace(0, 2 * np.pi, 24)
        rr = arm_len * 0.18
        ring = tip[None, :] + rr * (np.cos(th)[:, None] * R[:, 0][None, :]
                                    + np.sin(th)[:, None] * R[:, 1][None, :])
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2],
                color="dimgray", lw=0.8, zorder=zorder)


def camera_axis(q):
    qw, qx, qy, qz = np.asarray(q, dtype=float)
    bz_x = 2.0 * (qx * qz + qw * qy)
    bz_y = 2.0 * (qy * qz - qw * qx)
    bz_z = qw * qw - qx * qx - qy * qy + qz * qz
    cam = -np.array([bz_x, bz_y, bz_z], dtype=float)
    return cam / max(float(np.linalg.norm(cam)), 1e-12)


def set_3d_limits(ax, dpos, target=np.zeros(3), extra=None, center=None):
    """Center the chosen point in XY and keep Z visually compact.

    `center` overrides the XY center (default: final target). In the moving
    case we pass the midpoint of the target trajectory so the target's path
    stays centered as it sweeps through the scene.
    """
    target = np.asarray(target, dtype=float)
    pts = dpos if extra is None else np.vstack([dpos, extra])
    if center is None:
        center_xy = target[:2]
    else:
        center_xy = np.asarray(center, dtype=float)[:2]
    xy_rel = pts[:, :2] - center_xy
    r_xy = max(float(np.nanmax(np.abs(xy_rel))), anim.TARGET_HALF * 1.45, 0.75)
    r_xy *= 1.12
    z_hi = max(float(np.nanmax(pts[:, 2])) + 0.20, target[2] + 1.25)
    lo = np.array([center_xy[0] - r_xy, center_xy[1] - r_xy, 0.0])
    hi = np.array([center_xy[0] + r_xy, center_xy[1] + r_xy, z_hi])
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    return lo, hi


def draw_trajectory_3d(ax, dpos, quat, speed, bounds,
                       target_path=None, world_frame=False,
                       trigger_idx=None, draw_wall_projections=True,
                       draw_cone_silhouette=True,
                       draw_glideslope=True,
                       num_drone_samples=None,
                       drone_scale=0.30,
                       target_scale_pad=0.18,
                       draw_landing_projections=True,
                       trigger_markers=(),
                       los_cone_idx=None,
                       draw_los_cone=False):
    anim.style_3d(ax, "")
    ax.tick_params(labelsize=15)
    if target_path is None or not len(target_path):
        target_path = np.zeros_like(dpos)
    target = np.asarray(target_path[-1], dtype=float)
    extra = target_path if world_frame else None
    # In world frame, anchor the XY center to the MIDPOINT of the moving
    # target's path so the target's motion is centered in the view.
    if world_frame and len(target_path) > 1:
        center = np.array([float(np.mean(target_path[:, 0])),
                           float(np.mean(target_path[:, 1])),
                           0.0])
    else:
        center = None
    lo, hi = set_3d_limits(ax, dpos, target, extra=extra, center=center)
    extent = float(np.max(hi - lo))

    # Speed-colored trajectory.
    pts = dpos.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(
        segs, cmap=plt.cm.rainbow,
        norm=mcolors.Normalize(0.0, max(float(np.nanmax(speed)), 1e-6)),
        array=speed[:-1], lw=2.8, zorder=5,
    )
    ax.add_collection3d(lc)

    # Back-wall projections only — XZ wall at y=hi and YZ wall at x=lo.
    # The floor (XY) projection is intentionally omitted.
    n = len(dpos)
    if draw_wall_projections:
        ax.plot(dpos[:, 0], np.full(n, hi[1]), dpos[:, 2],
                color=C_PROJ, lw=0.9, alpha=0.55, zorder=0,
                label="Trajectory Projections")
        ax.plot(np.full(n, lo[0]), dpos[:, 1], dpos[:, 2],
                color=C_PROJ, lw=0.9, alpha=0.55, zorder=0)

    if not trigger_markers and trigger_idx is not None:
        trigger_markers = (("$a_{\\mathrm{land}}$", trigger_idx, C_TRIG_ALT),)

    # Trigger markers — short horizontal dashed segments on the two back-wall
    # projections plus a small point on the trajectory. The line height is the
    # actual logged sample where the corresponding trigger event occurs.
    z_trig = None
    for im, (label, marker_idx, color) in enumerate(trigger_markers):
        if marker_idx is None or not np.isfinite(marker_idx):
            continue
        ti = int(marker_idx)
        if ti < 0 or ti >= n:
            continue
        z_mark = float(dpos[ti, 2])
        if label == "$a_{\\mathrm{land}}$":
            z_trig = z_mark
        xtr, ytr = float(dpos[ti, 0]), float(dpos[ti, 1])
        seg_x = 0.10 * float(hi[0] - lo[0])
        seg_y = 0.10 * float(hi[1] - lo[1])
        dash_offset = 2 * im
        ax.plot([xtr - seg_x, xtr + seg_x], [hi[1], hi[1]], [z_mark, z_mark],
                color=color, linestyle=(dash_offset, (5, 3)), lw=1.2, zorder=2,
                label=label)
        ax.plot([lo[0], lo[0]], [ytr - seg_y, ytr + seg_y], [z_mark, z_mark],
                color=color, linestyle=(dash_offset, (5, 3)), lw=1.2, zorder=2)
        ax.scatter([xtr], [ytr], [z_mark], s=24, c=color,
                   depthshade=False, zorder=16)

    # Cone silhouette on the back walls — truncated to the trigger altitude
    # so the wall lines match the height of the trigger crossing point.
    if draw_cone_silhouette:
        # Match the actual 3-D glideslope cone exactly: same apex, same
        # height, same half-angle. The wireframe cone is drawn below with
        # apex = target + (0, 0, -gs_apex_offset) and height CONE_H_eff.
        target_xy = np.asarray(target_path[-1], dtype=float)[:2] if target_path is not None and len(target_path) else np.array([0.0, 0.0])
        target_z = float(target[2])
        apex_z = target_z - float(bounds["gs_apex_offset"])
        tan_a = float(bounds.get("gs_post_tan", np.tan(np.radians(float(bounds["gs_post_deg"])))))
        cone_h_eff = min(max(float(bounds["alt_trig"]) + bounds["gs_apex_offset"], 0.85), 1.15)
        z_top = apex_z + cone_h_eff
        r = tan_a * cone_h_eff
        # XZ-wall silhouette (y = hi[1]).
        ax.plot([target_xy[0] - r, target_xy[0], target_xy[0] + r],
                [hi[1], hi[1], hi[1]],
                [z_top, apex_z, z_top],
                color=C_CONE, ls="dotted", lw=1.0, zorder=0)
        # YZ-wall silhouette (x = lo[0]).
        ax.plot([lo[0], lo[0], lo[0]],
                [target_xy[1] - r, target_xy[1], target_xy[1] + r],
                [z_top, apex_z, z_top],
                color=C_CONE, ls="dotted", lw=1.0, zorder=0)

    if world_frame and target_path is not None and len(target_path):
        ax.plot(target_path[:, 0], target_path[:, 1], target_path[:, 2],
                color=anim.TARGET, lw=1.4, alpha=0.85)
        # Direction arrows along the target path so the reader can read the
        # target's heading at a glance.
        nT = len(target_path)
        arrow_idx = np.unique(np.linspace(2, nT - 2, 4, dtype=int))
        for k in arrow_idx:
            if k < 1 or k >= nT - 1:
                continue
            p = target_path[k]
            d = target_path[k + 1] - target_path[k - 1]
            nrm = float(np.linalg.norm(d))
            if nrm < 1e-9:
                continue
            d = d / nrm * 0.04 * extent
            ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2],
                      color=anim.TARGET, lw=1.0, arrow_length_ratio=0.45,
                      zorder=3)

    # Landing point projected onto the two back walls.
    if draw_landing_projections:
        tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
        ax.scatter([tx], [hi[1]],  [tz],    s=24, c=anim.TARGET, depthshade=False, zorder=2)
        ax.scatter([lo[0]], [ty],  [tz],    s=24, c=anim.TARGET, depthshade=False, zorder=2,
                   label="Landing point")

    # Final active glideslope only. The STC is ||r_xy|| <= tan(alpha)(z + z0),
    # where alpha is the cone half-angle from vertical.
    if draw_glideslope:
        anim.GS_APEX_OFFSET = float(bounds["gs_apex_offset"])
        anim.GS_TAN = float(bounds.get("gs_post_tan", np.tan(np.radians(float(bounds["gs_post_deg"])))))
        old_h = anim.CONE_H
        old_cone = anim.CONE
        anim.CONE_H = min(max(float(bounds["alt_trig"]) + bounds["gs_apex_offset"], 0.85), 1.15)
        anim.CONE = C_CONE
        anim.draw_constraint(ax, "gs", target, alpha=0.42)
        # Legend proxy for the cone (animate.draw_constraint draws unlabeled lines).
        ax.plot([], [], color=anim.CONE, lw=1.3, label="Glideslope cone")
        anim.CONE_H = old_h
        anim.CONE = old_cone

    if draw_los_cone and los_cone_idx is not None and np.isfinite(los_cone_idx):
        li = int(los_cone_idx)
        if 0 <= li < n:
            los_center = np.asarray(dpos[li], dtype=float)
            los_dir = camera_axis(quat[li])
            ground_z = float(target[2]) if world_frame else 0.0
            if abs(los_dir[2]) > 1e-9:
                los_h = abs(float(los_center[2] - ground_z) / float(los_dir[2]))
            else:
                los_h = float(np.linalg.norm(los_center - target))
            los_h = float(np.clip(los_h, 0.18, min(max(0.28, 0.12 * extent), 0.55)))
            if np.linalg.norm(np.cross([0.0, 0.0, 1.0], los_dir)) < 1e-9 and los_dir[2] < 0.0:
                los_dir = np.array([1e-6, 0.0, -1.0])
            plot_cone_wireframe(
                ax, los_center, los_dir,
                angle_deg=float(bounds["los_cone_deg"]), height=los_h,
                cc=C_TRIG_THR, lw=0.42, zorder=8, num_points=34,
            )
            ax.plot([], [], color=C_TRIG_THR, lw=1.2, linestyle="dotted",
                    label="LOS cone")

    # Use the animate.py target model, scaled down for a paper trajectory plot.
    old_vertex_defaults = anim.target_pad_vertices.__defaults__
    old_point_defaults = anim.target_pad_points.__defaults__
    old_wheel_defaults = anim.target_wheel_points.__defaults__
    old_wheel_radius = anim.WHEEL_RADIUS
    old_wheel_z = anim.WHEEL_Z_OFFSET
    try:
        # Smaller pad, smaller wheels, wheels closer to the box base.
        anim.target_pad_vertices.__defaults__ = (0.0, target_scale_pad)
        anim.target_pad_points.__defaults__ = (0.0, target_scale_pad, 0.008)
        anim.target_wheel_points.__defaults__ = (0.0, target_scale_pad)
        anim.WHEEL_RADIUS = 0.032 * (target_scale_pad / 0.18)
        anim.WHEEL_Z_OFFSET = -0.020 * (target_scale_pad / 0.18)

        if world_frame and len(target_path) > 1:
            # Show the target at multiple times so the reader can SEE it
            # moving. Pair each target snapshot with the drone snapshot at
            # the same time index so drone/target are visually grouped.
            nT = len(target_path)
            tgt_idxs = np.unique(np.linspace(0, nT - 1, 5, dtype=int))
            for j, kT in enumerate(tgt_idxs):
                is_final = j == len(tgt_idxs) - 1
                a = 0.85 if is_final else 0.25 + 0.12 * j
                yaw = 0.0
                if 0 < kT < nT - 1:
                    d = target_path[kT + 1] - target_path[kT - 1]
                    if np.linalg.norm(d[:2]) > 1e-9:
                        yaw = float(np.arctan2(d[1], d[0]))
                anim.draw_target_pad(ax, target_path[kT], yaw=yaw, alpha=a)
        else:
            anim.draw_target_pad(ax, target, yaw=0.0, alpha=0.58)
    finally:
        anim.target_pad_vertices.__defaults__ = old_vertex_defaults
        anim.target_pad_points.__defaults__ = old_point_defaults
        anim.target_wheel_points.__defaults__ = old_wheel_defaults
        anim.WHEEL_RADIUS = old_wheel_radius
        anim.WHEEL_Z_OFFSET = old_wheel_z

    # Drone snapshots — pair the moving case with the target snapshots so
    # each rendered drone has an associated rendered target.
    if num_drone_samples is not None:
        base = list(np.unique(np.linspace(0, n - 1, num_drone_samples, dtype=int)))
    elif world_frame and len(target_path) > 1:
        nT = len(target_path)
        base = list(np.unique(np.linspace(0, nT - 1, 10, dtype=int)))
    else:
        base = list(np.linspace(0, n - 1, 7, dtype=int))
    if trigger_idx is not None and 0 < int(trigger_idx) < n - 1:
        ti = int(trigger_idx)
        base.extend(int(round(ti * f)) for f in (0.30, 0.60))
    idxs = sorted(set(int(k) for k in base if 0 <= int(k) < n))
    last = idxs[-1] if idxs else None
    for k in idxs:
        pos = dpos[k].copy()
        # The final-node position can dip slightly below z=0 in the data;
        # render the drone sitting on the pad rather than under it.
        if k == last:
            pos[2] = max(float(pos[2]), 0.04)
        anim.draw_drone(ax, pos, quat[k], scale=drone_scale, alpha=0.38)

    if world_frame:
        ax.set_xlabel("$x$ [m]", fontsize=17, labelpad=10)
        ax.set_ylabel("$y$ [m]", fontsize=17, labelpad=10)
        ax.set_zlabel("$z$ [m]", fontsize=17, labelpad=20)
    else:
        ax.set_xlabel("$r_x$ [m]", fontsize=17, labelpad=10)
        ax.set_ylabel("$r_y$ [m]", fontsize=17, labelpad=10)
        ax.set_zlabel("$r_z$ [m]", fontsize=17, labelpad=20)
    # To make the plots look like they have the same dimensions in the paper,
    # use a consistent visual box aspect regardless of the physical units range.
    try:
        ax.set_box_aspect((1, 1, 0.85))
    except Exception:
        pass
    ax.view_init(elev=13, azim=-58)
    return lc


# ── State panel, notebook style. ─────────────────────────────────────────────
def first_trigger_time(t, y, eps=1e-9):
    idx = np.flatnonzero(np.asarray(y, dtype=float) > eps)
    return float(t[idx[0]]) if len(idx) else None


def first_trigger_index(y, eps=1e-9):
    idx = np.flatnonzero(np.asarray(y, dtype=float) > eps)
    return int(idx[0]) if len(idx) else None


def draw_trigger_lines(ax, triggers, with_labels=False):
    for i, (label, tt, color) in enumerate(triggers):
        if tt is None or not np.isfinite(tt):
            continue
        ax.axvline(tt, c=color, linestyle=(2 * i, (5, 3)), lw=1.35,
                   label=label if with_labels else None)


def sampled_indices(n, max_nodes=18):
    if n <= max_nodes:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_nodes, dtype=int))


def panel_constraint(ax, t, y, ylabel, switch_t,
                     upper_pre=None, upper_post=None,
                     lower_pre=None, lower_post=None,
                     ymin=None, ymax=None,
                     triggers=(),
                     label_state="State", trigger_label=False):
    have_sw = switch_t is not None and np.isfinite(switch_t)
    t_end = float(t[-1])
    t_sw = float(switch_t) if have_sw else t_end

    # Bounds — pre-switch segment.
    if upper_pre is not None:
        ax.plot([t[0], t_sw], [upper_pre, upper_pre],
                c=C_UPPER, linestyle=(0, (5, 3)), lw=1.35, label="Upper bound")
    if lower_pre is not None:
        ax.plot([t[0], t_sw], [lower_pre, lower_pre],
                c=C_LOWER, linestyle=(0, (5, 3)), lw=1.35, label="Lower bound")

    # Trigger line + post-switch tightened bounds.
    if have_sw:
        draw_trigger_lines(ax, triggers, with_labels=trigger_label)
        if upper_post is not None:
            ax.plot([t_sw, t_end], [upper_post, upper_post],
                    c=C_UPPER, linestyle=(0, (5, 3)), lw=1.35)
        if lower_post is not None:
            ax.plot([t_sw, t_end], [lower_post, lower_post],
                    c=C_LOWER, linestyle=(0, (5, 3)), lw=1.35)

    # State trace only — discrete optimization nodes have no reader value here.
    ax.plot(t, y, c=C_STATE, lw=1.5, label=label_state)

    ax.set_xlim(t[0], t_end)
    if ymin is None or ymax is None:
        vals = [y]
        for v in (upper_pre, upper_post, lower_pre, lower_post):
            if v is not None:
                vals.append(np.array([v]))
        flat = np.concatenate([np.atleast_1d(v).reshape(-1) for v in vals])
        finite = flat[np.isfinite(flat)]
        if len(finite):
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            pad = max((hi - lo) * 0.12, 0.05)
            ymin = ymin if ymin is not None else lo - pad
            ymax = ymax if ymax is not None else hi + pad
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    ax.set_ylabel(ylabel, fontsize=Y_FS, labelpad=2)
    ax.yaxis.set_label_coords(-0.18, 0.5)
    ax.tick_params(labelsize=14)
    ax.grid(True, color="0.88", lw=0.45, ls="--", alpha=0.55)
    # Always show y-ticks as decimals (e.g. 8.0 instead of 8) so the axes
    # remain visually consistent across panels.
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))


def panel_trigger_signal(ax, t, a_land, a_thrust, triggers):
    ax.plot(t, a_land, c=C_TRIG_ALT, lw=1.35, label="$a_{\\mathrm{land}}$")
    ax.plot(t, a_thrust, c=C_TRIG_THR, lw=1.35, label="$a_T$")
    draw_trigger_lines(ax, triggers, with_labels=True)
    ax.set_xlim(t[0], t[-1])
    vals = np.concatenate([np.asarray(a_land), np.asarray(a_thrust)])
    ymax = max(float(np.nanmax(vals)) * 1.18, 1e-3)
    ax.set_ylim(-0.02 * ymax, ymax)
    ax.set_ylabel("Trigger signals", fontsize=Y_FS, labelpad=2)
    ax.yaxis.set_label_coords(-0.18, 0.5)
    ax.tick_params(labelsize=14)
    ax.grid(True, color="0.88", lw=0.45, ls="--", alpha=0.55)
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=12, loc="best", framealpha=0.85)


def stage_constraint_end_time(t, a_stage):
    active = np.asarray(a_stage, dtype=float) > 1e-9
    if np.any(active):
        end_idx = int(np.flatnonzero(active)[-1])
        return float(t[end_idx])
    return None


def stage_constraint_end_index(a_stage):
    active = np.asarray(a_stage, dtype=float) > 1e-9
    if np.any(active):
        return int(np.flatnonzero(active)[-1])
    return None


def compute_los_angle_deg(rxyz, quat):
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    bz_x = 2.0 * (qx * qz + qw * qy)
    bz_y = 2.0 * (qy * qz - qw * qx)
    bz_z = qw * qw - qx * qx - qy * qy + qz * qz
    cam = -np.column_stack([bz_x, bz_y, bz_z])
    los = -rxyz
    axial = np.einsum("ij,ij->i", los, cam)
    los_norm2 = np.einsum("ij,ij->i", los, los)
    lat2 = np.maximum(0.0, los_norm2 - axial * axial)
    lateral = np.sqrt(lat2 + 1e-12)
    return np.degrees(np.arctan2(lateral, axial))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",
                    default="benchmark/results/quad_cf_landing_single_horizon_stc")
    ap.add_argument("--csv", default=None,
                    help="CSV inside --dir. Defaults to single_horizon_stc.csv or moving_executed.csv.")
    ap.add_argument("--kind", default="auto",
                    choices=("auto", "single", "single_horizon_stc", "stateswitch_stc"),
                    help="Dataset type. Auto-detected from --dir/--csv by default.")
    ap.add_argument("--out", default="stc_paper.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    base = resolve_result_dir(args.dir)
    kind, csv_path = detect_dataset(base, args.csv, args.kind)
    df, data = read_and_normalize_dataset(base, csv_path, kind)

    b = load_current_bounds(kind, base)
    t = data["t"]
    rxyz = data["rxyz"]
    rvxyz = data["rvxyz"]
    world_xyz = data["world_xyz"]
    target_xyz = data["target_xyz"]
    quat = data["quat"]
    omega = data["omega"]
    fz = data["fz"]

    speed = (df["rel_speed"].to_numpy(dtype=float)
             if "rel_speed" in df.columns
             else df["stc_rel_speed"].to_numpy(dtype=float)
             if "stc_rel_speed" in df.columns
             else np.linalg.norm(rvxyz, axis=1))
    omega_n = (df["omega_norm"].to_numpy(dtype=float)
               if "omega_norm" in df.columns
               else df["stc_omega_norm"].to_numpy(dtype=float)
               if "stc_omega_norm" in df.columns
               else np.linalg.norm(omega, axis=1))

    rxy = np.linalg.norm(rxyz[:, :2], axis=1)
    dz = rxyz[:, 2] + b["gs_apex_offset"]
    # Glideslope angle from vertical, in degrees. Constraint: gs_angle <= alpha.
    gs_angle = np.degrees(np.arctan2(rxy, np.maximum(dz, 1e-9)))
    q_perp = (df["tilt_sin"].to_numpy(dtype=float)
              if "tilt_sin" in df.columns
              else df["stc_tilt_sin"].to_numpy(dtype=float)
              if "stc_tilt_sin" in df.columns
              else np.linalg.norm(quat[:, 1:3], axis=1))
    tilt_deg = 2.0 * np.degrees(np.arcsin(np.clip(q_perp, 0.0, 1.0)))
    a_land = (df["ctcs_a_altitude_state"].to_numpy(dtype=float)
              if "ctcs_a_altitude_state" in df.columns
              else df.get("trig_env", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float))
    if "ctcs_a_stage" in df.columns:
        a_stage = df["ctcs_a_stage"].to_numpy(dtype=float)
    else:
        sigma_land = (df["sigma_land"].to_numpy(dtype=float)
                      if "sigma_land" in df.columns
                      else np.ones(len(df), dtype=float))
        a_stage = 1.0 - np.clip(sigma_land, 0.0, 1.0)
    a_los_margin = (b["los_alt_trig"] - rxyz[:, 2]) / b["los_trigger_scale"]
    a_los_geom = np.where(a_los_margin > 0.0, b["los_trigger_floor"] + a_los_margin, 0.0)
    if "ctcs_a_los" in df.columns:
        a_los_csv = df["ctcs_a_los"].to_numpy(dtype=float)
        # STC-off runs zero the CT-cSTC trigger export; for ON/OFF comparison
        # plots, still draw the geometric trigger where the LOS limit would bind.
        a_los = a_los_csv if np.nanmax(a_los_csv) > 1e-12 else a_los_geom
    else:
        a_los = a_los_geom
    los_angle = compute_los_angle_deg(rxyz, quat)
    land_t = first_trigger_time(t, a_land)
    los_t = first_trigger_time(t, a_los)
    land_idx = first_trigger_index(a_land)
    los_idx = first_trigger_index(a_los)
    trig_land   = (("$a_{\\mathrm{land}}$", land_t, C_TRIG_ALT),)
    trig_los    = (("$a_{\\mathrm{los}}$", los_t, C_TRIG_THR),)
    stage_t = stage_constraint_end_time(t, a_stage)
    stage_idx = stage_constraint_end_index(a_stage)
    trig_stage = (("$a_{\\mathrm{stage}}$", stage_t, C_TRIG_STAGE),)
    trigger_markers_3d = (
        ("$a_{\\mathrm{stage}}$", stage_idx, C_TRIG_STAGE),
        ("$a_{\\mathrm{los}}$", los_idx, C_TRIG_THR),
        ("$a_{\\mathrm{land}}$", land_idx, C_TRIG_ALT),
    )

    # ── Layout knobs (edit these to retune EACH figure) ────────────────────
    # All x/y/w/h values are FIGURE FRACTIONS in [0, 1].

    # 3-D figure (Fig. 1 in the reference paper).
    # If world-frame is active, we use a wide figure with two subplots.
    TWIN_3D          = kind == "stateswitch_stc"
    # Match width to the state figure (11.0) so paper figures line up.
    # Reserve top band for the legend so it isn't cropped, and reserve right
    # edge for the colorbar. Z labels need bigger labelpad to clear the box.
    FIG3D_W, FIG3D_H = (18.0, 7.0) if TWIN_3D else (6.5, 6.4)
    if TWIN_3D:
        # Wider canvas (18) so the legend can fit in one row at fontsize 22.
        # Content (plots + colorbar) shifted right so there's equal margin on
        # both sides — gives the legend room to extend right without clipping.
        AX3D_BOX_L   = (0.128, 0.060, 0.320, 0.820)   # World frame   (x: 0.128–0.448)
        AX3D_BOX_R   = (0.488, 0.060, 0.320, 0.820)   # Relative frame (x: 0.488–0.808)
        CBAR_BOX     = (0.863, 0.190, 0.010, 0.500)   # right of right panel
        LEG3D_BBOX   = (0.128, 0.880, 0.760, 0.115)   # spans from left plot to right of colorbar
    else:
        AX3D_BOX     = (0.000, 0.000, 0.770, 0.840)
        CBAR_BOX     = (0.860, 0.180, 0.024, 0.640)
        LEG3D_BOX    = (0.02, 1.17)                   # bbox in 3-D axes coords

    # State figure (Fig. 2 in the reference paper) — two rows only.
    # Width = 16 so the legend fits 7 items in one row at fontsize 18; height
    # = 8.5 and tall panels so the rotated y-labels (Y_FS=17) fit vertically.
    FIGS_W, FIGS_H   = 16.0, 8.5
    STATE_PANEL_W    = 0.220
    STATE_PANEL_H    = 0.350
    STATE_LEFT       = 0.080
    STATE_COL_GAP    = 0.090
    STATE_ROW_Y      = (0.480, 0.080)
    # Figure-level legend: bbox is (x, y, width, height) so it spans the
    # full state-panel band when used with mode="expand". The y value sits
    # just above the top row so there is minimal whitespace between the
    # legend strip and the plots.
    LEGS_BBOX        = (STATE_LEFT, 0.850,
                        3 * STATE_PANEL_W + 2 * STATE_COL_GAP, 0.060)
    LEGS_NCOL        = 8
    LEG_STATE_FS     = 18

    world_frame_3d = kind == "stateswitch_stc"
    plot_xyz = world_xyz if world_frame_3d else rxyz

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1 — 3-D trajectory.
    # ════════════════════════════════════════════════════════════════════
    fig3d = plt.figure(figsize=(FIG3D_W, FIG3D_H), facecolor="white")
    if TWIN_3D:
        DSCALE = 0.45
        TSCALE = 0.30

        # Left panel: World Frame (No Glideslope, Reasonable Drone Samples)
        ax3dL = fig3d.add_axes(list(AX3D_BOX_L), projection="3d", computed_zorder=False)
        ax3dL.set_title("World Space", fontsize=X_FS, pad=0)
        lc = draw_trajectory_3d(ax3dL, world_xyz, quat, speed, b,
                                target_path=target_xyz,
                                world_frame=True,
                                draw_cone_silhouette=False,
                                draw_glideslope=False,
                                num_drone_samples=9,
                                drone_scale=DSCALE,
                                target_scale_pad=TSCALE,
                                draw_landing_projections=True,
                                trigger_markers=trigger_markers_3d)

        # Right panel: Relative Frame (With Glideslope, Single-Horizon look)
        ax3dR = fig3d.add_axes(list(AX3D_BOX_R), projection="3d", computed_zorder=False)
        ax3dR.set_title("Relative Space", fontsize=X_FS, pad=10)
        draw_trajectory_3d(ax3dR, rxyz, quat, speed, b,
                           target_path=np.zeros_like(rxyz),
                           world_frame=False,
                           draw_cone_silhouette=True,
                           draw_glideslope=True,
                           num_drone_samples=6,
                           drone_scale=DSCALE,
                           target_scale_pad=TSCALE,
                           draw_landing_projections=True,
                           trigger_markers=trigger_markers_3d,
                           los_cone_idx=los_idx,
                           draw_los_cone=True)
        ax3ds = [ax3dL, ax3dR]
    else:
        ax3d = fig3d.add_axes(list(AX3D_BOX), projection="3d", computed_zorder=False)
        lc = draw_trajectory_3d(ax3d, plot_xyz, quat, speed, b,
                                target_path=target_xyz if world_frame_3d else None,
                                world_frame=world_frame_3d,
                                trigger_markers=trigger_markers_3d)
        ax3ds = [ax3d]

    cax = fig3d.add_axes(list(CBAR_BOX))
    cbar = fig3d.colorbar(lc, cax=cax)
    cbar.set_label("Speed, $\\|v\\|_2$ [m s$^{-1}$]", fontsize=15, labelpad=6)
    cbar.ax.tick_params(labelsize=14)

    # 3-D legends.
    if TWIN_3D:
        # Collect all unique labels from both panels.
        all_handles, all_labels = [], []
        for ax in ax3ds:
            h, l = ax.get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)
        seen = set()
        pairs = []
        for h, l in zip(all_handles, all_labels):
            if l and l not in seen:
                seen.add(l)
                pairs.append((h, l))
        if pairs:
            # Single row, items packed tightly (no mode=expand to avoid wide
            # gaps between entries). Centered horizontally in the strip.
            # Anchor legend to the LEFT edge of the left 3D panel so it can't
            # spill off the canvas (legend is wider than the two-panel span at
            # fontsize 22, so true centering on the plots would clip the left).
            fig3d.legend([p[0] for p in pairs], [p[1] for p in pairs],
                         loc="upper left",
                         bbox_to_anchor=(AX3D_BOX_L[0], LEG3D_BBOX[1] + LEG3D_BBOX[3]),
                         ncol=len(pairs),
                         fontsize=18, frameon=True, framealpha=0.92, edgecolor="0.7",
                         handlelength=1.7, columnspacing=0.9, borderpad=0.6)
    else:
        for ax in ax3ds:
            h3, l3 = ax.get_legend_handles_labels()
            seen = set()
            pairs = []
            for h, l in zip(h3, l3):
                if l and l not in seen:
                    seen.add(l)
                    pairs.append((h, l))
            if pairs:
                ax.legend([p[0] for p in pairs], [p[1] for p in pairs],
                            loc="upper left", bbox_to_anchor=LEG3D_BOX,
                            bbox_transform=ax.transAxes,
                            fontsize=LEG_FS, framealpha=0.92, edgecolor="0.7",
                            handlelength=2.4, labelspacing=0.45, borderpad=0.5)

    out_arg = Path(args.out)
    out_stem = out_arg.with_suffix("")
    suffix = out_arg.suffix if out_arg.suffix else ".png"
    out_3d = (Path.cwd() / out_stem) if not out_stem.is_absolute() else out_stem
    out_3d = out_3d.parent / f"{out_3d.name}_3d{suffix}"
    out_3d.parent.mkdir(parents=True, exist_ok=True)
    fig3d.savefig(out_3d, dpi=args.dpi, facecolor="white")
    print(f"wrote {out_3d}")

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 2 — state and control panels.
    # ════════════════════════════════════════════════════════════════════
    figS = plt.figure(figsize=(FIGS_W, FIGS_H), facecolor="white")
    state_w, state_h = STATE_PANEL_W, STATE_PANEL_H
    x0s = [STATE_LEFT + i * (STATE_PANEL_W + STATE_COL_GAP) for i in range(3)]
    y_r1, y_r2 = STATE_ROW_Y

    # Row 1: speed, LOS cone angle, glideslope angle.
    # Speed/glideslope/omega/tilt panels use a_land. LOS uses a_los.
    ax_v = figS.add_axes([x0s[0], y_r1, state_w, state_h])
    panel_constraint(ax_v, t, speed,
                     ylabel="Speed, $v=\\|\\dot r\\|_2$ [m/s]",
                     switch_t=land_t,
                     upper_pre=b["v_pre"], upper_post=b["v_post"],
                     ymin=0.0, triggers=trig_land, trigger_label=True)

    ax_los = figS.add_axes([x0s[1], y_r1, state_w, state_h])
    panel_constraint(ax_los, t, los_angle,
                     ylabel="LOS cone angle, $\\theta_{los}$ [deg]",
                     switch_t=los_t,
                     upper_post=b["los_cone_deg"],
                     ymin=0.0,
                     triggers=trig_los,
                     trigger_label=True)

    ax_gs = figS.add_axes([x0s[2], y_r1, state_w, state_h])
    panel_constraint(ax_gs, t, gs_angle,
                     ylabel="Glideslope angle, $\\theta_{gs}$ [deg]",
                     switch_t=land_t,
                     upper_post=b["gs_post_deg"],  # only post-trigger cone is enforced
                     ymin=0.0, triggers=trig_land)

    # Row 2: angular rate, tilt angle, staging altitude.
    ax_om = figS.add_axes([x0s[0], y_r2, state_w, state_h])
    panel_constraint(ax_om, t, omega_n,
                     ylabel="Angular rate, $\\|\\omega\\|_2$ [rad/s]",
                     switch_t=land_t,
                     upper_pre=b["omega_pre"], upper_post=b["omega_post"],
                     ymin=0.0, triggers=trig_land)

    ax_tilt = figS.add_axes([x0s[1], y_r2, state_w, state_h])
    panel_constraint(ax_tilt, t, tilt_deg,
                     ylabel="Tilt, $\\theta$ [deg]",
                     switch_t=land_t,
                     upper_pre=b["tilt_pre_deg"], upper_post=b["tilt_post_deg"],
                     ymin=0.0, triggers=trig_land)

    ax_trig = figS.add_axes([x0s[2], y_r2, state_w, state_h])
    panel_constraint(ax_trig, t, rxyz[:, 2],
                     ylabel="Stage altitude, $z$ [m]",
                     switch_t=stage_t,
                     lower_pre=b["z_stage"],
                     ymin=0.0,
                     triggers=trig_stage,
                     trigger_label=True)
    # Drop the per-axis legend — the figure-level legend covers these entries.
    leg = ax_trig.get_legend()
    if leg is not None:
        leg.remove()

    for ax in (ax_v, ax_los, ax_gs):
        ax.set_xlabel("")
    for ax in (ax_om, ax_tilt, ax_trig):
        ax.set_xlabel("Time [s]", fontsize=X_FS)

    # ── State-figure legend strip (framed box above the panel grid) ────────
    pool: dict[str, "Line2D"] = {}
    for ax in figS.axes:
        hs, ls = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls):
            if l and l not in pool:
                pool[l] = h
    desired_order = [
        "State",
        "Upper bound", "Lower bound",
        "$a_{\\mathrm{los}}$",
        "$a_{\\mathrm{land}}$",
        "$a_{\\mathrm{stage}}$",
    ]
    lines, labels = [], []
    for key in desired_order:
        if key in pool:
            h = pool.pop(key)
            lines.append(h)
            labels.append(h.get_label())

    # Full-width framed legend across the entire state-panel band.
    figS.legend(lines, labels, loc="lower left",
                ncol=LEGS_NCOL, fontsize=LEG_STATE_FS,
                bbox_to_anchor=LEGS_BBOX, mode="expand",
                frameon=True, framealpha=0.92, edgecolor="0.7",
                handlelength=1.8, columnspacing=1.0, borderpad=0.5)
    figS.align_ylabels([ax_v, ax_om])
    figS.align_ylabels([ax_los, ax_tilt])
    figS.align_ylabels([ax_gs, ax_trig])

    out_st = (Path.cwd() / out_stem) if not out_stem.is_absolute() else out_stem
    out_st = out_st.parent / f"{out_st.name}_states{suffix}"
    figS.savefig(out_st, dpi=args.dpi, facecolor="white")
    print(f"wrote {out_st}")
    print(f"dataset={kind}  csv={csv_path}")


if __name__ == "__main__":
    main()
