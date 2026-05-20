#!/usr/bin/env python3
"""Generate a bag-first CoManDO analysis PDF.

The rosbag is treated as physical truth. CSV files are used only for solver
internals such as accepted OCP nodes, solve timing, and solver events.
"""

import argparse
import os
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")

FIG_BG = "#f1f1f1"
PANEL_BG = "#e9e9e9"
GRID_COL = "#a5a5a5"
SPINE_COL = "#303848"
TEXT_COL = "#000000"
C_DRONE = "#2388d9"
C_TARGET = "#d12be6"
C_PLAN = "#f0a020"
C_ERR = "#d84b4b"
C_CMD = "#2ca25f"
COMP_COLS = ["#d84b4b", "#2ca25f", "#2388d9", "#9467bd"]
AZ_VIEWS = [35, 125, 215, 305]
PAGE_WIDE = (12.8, 7.2)
PAGE_TALL = (12.8, 9.0)
PAGE_GRID = (13.2, 9.4)


def _fig(title, figsize=PAGE_WIDE):
    fig = plt.figure(figsize=figsize, facecolor=FIG_BG, constrained_layout=False)
    fig.suptitle(title, color=TEXT_COL, fontsize=13, fontweight="bold", y=0.98)
    return fig


def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(PANEL_BG)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=TEXT_COL, fontsize=13, fontweight="bold")
    ax.grid(True, color=GRID_COL, linewidth=0.8, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COL)
        sp.set_linewidth(1.1)


def _style_3d(ax, title):
    ax.set_facecolor(FIG_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=12, fontweight="bold")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    for attr in ("xaxis", "yaxis", "zaxis"):
        getattr(ax, attr)._axinfo["grid"]["color"] = GRID_COL


def stamp_sec(msg, bag_time_ns):
    hdr = getattr(msg, "header", None)
    if hdr is not None:
        sec = getattr(hdr.stamp, "sec", 0)
        nsec = getattr(hdr.stamp, "nanosec", 0)
        if sec != 0 or nsec != 0:
            return float(sec) + float(nsec) * 1e-9
    return float(bag_time_ns) * 1e-9


def quat_to_rotmat(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-12
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)],
        [2 * (qx*qy + qw*qz), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qw*qx)],
        [2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx*qx + qy*qy)],
    ])


def read_bag(bag_dir):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "Could not import ROS bag Python modules. Source ROS and the workspace first:\n"
            "  source /opt/ros/humble/setup.zsh\n"
            "  source install/setup.zsh"
        ) from exc

    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage, converter)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_types = {topic: get_message(type_name) for topic, type_name in topic_types.items()}

    records = {
        "drone_pose": [],
        "drone_odom": [],
        "target_odom": [],
        "target_frame_odom": [],
        "cmd_hover": [],
        "cmd_full_state": [],
        "planned_paths": [],
    }
    topic_counts = {topic: 0 for topic in topic_types}

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        msg_type = msg_types.get(topic)
        if msg_type is None:
            continue
        msg = deserialize_message(data, msg_type)
        t = stamp_sec(msg, t_ns)

        if topic.endswith("/pose"):
            p = msg.pose.position
            q = msg.pose.orientation
            records["drone_pose"].append({
                "t": t, "x": p.x, "y": p.y, "z": p.z,
                "qw": q.w, "qx": q.x, "qy": q.y, "qz": q.z,
            })
        elif topic.endswith("/odom") and topic.startswith("/cf_"):
            records["drone_odom"].append(odom_row(msg, t, ""))
        elif topic == "/target/odom":
            records["target_odom"].append(odom_row(msg, t, "tgt_"))
        elif topic == "/drone/target_frame_odom":
            records["target_frame_odom"].append(odom_row(msg, t, "tf_"))
        elif topic.endswith("/cmd_hover"):
            records["cmd_hover"].append({
                "t": t, "vx": msg.vx, "vy": msg.vy,
                "z_distance": msg.z_distance, "yaw_rate": msg.yaw_rate,
            })
        elif topic.endswith("/cmd_full_state"):
            p = msg.pose.position
            q = msg.pose.orientation
            v = msg.twist.linear
            w = msg.twist.angular
            a = msg.acc
            records["cmd_full_state"].append({
                "t": t, "x": p.x, "y": p.y, "z": p.z,
                "qw": q.w, "qx": q.x, "qy": q.y, "qz": q.z,
                "vx": v.x, "vy": v.y, "vz": v.z,
                "wx": w.x, "wy": w.y, "wz": w.z,
                "ax": a.x, "ay": a.y, "az": a.z,
            })
        elif topic.endswith("/planned_trajectory"):
            frame_id = getattr(msg.header, "frame_id", "")
            pts = []
            for pose in msg.poses:
                p = pose.pose.position
                q = pose.pose.orientation
                pts.append([p.x, p.y, p.z, q.w, q.x, q.y, q.z])
            records["planned_paths"].append({
                "t": t,
                "frame_id": frame_id,
                "points": np.asarray(pts, dtype=float) if pts else np.empty((0, 7)),
            })

    return records, topic_counts, topic_types


def odom_row(msg, t, prefix):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    v = msg.twist.twist.linear
    w = msg.twist.twist.angular
    return {
        "t": t,
        f"{prefix}x": p.x, f"{prefix}y": p.y, f"{prefix}z": p.z,
        f"{prefix}qw": q.w, f"{prefix}qx": q.x, f"{prefix}qy": q.y, f"{prefix}qz": q.z,
        f"{prefix}vx": v.x, f"{prefix}vy": v.y, f"{prefix}vz": v.z,
        f"{prefix}wx": w.x, f"{prefix}wy": w.y, f"{prefix}wz": w.z,
    }


def df_from(records):
    df = pd.DataFrame(records)
    if not df.empty and "t" in df.columns:
        df = df.sort_values("t").drop_duplicates("t").reset_index(drop=True)
    return df


def normalize_times(dfs, paths):
    starts = []
    for df in dfs:
        if df is not None and not df.empty:
            starts.append(float(df["t"].iloc[0]))
    for path in paths:
        starts.append(float(path["t"]))
    t0 = min(starts) if starts else 0.0
    for df in dfs:
        if df is not None and not df.empty:
            df["time"] = df["t"] - t0
    for path in paths:
        path["time"] = path["t"] - t0
    return t0


def align_to(left, right, cols):
    if left.empty or right.empty:
        return pd.DataFrame(index=left.index)
    l = left[["t"]].copy().sort_values("t").reset_index()
    r = right[["t"] + cols].copy().sort_values("t")
    merged = pd.merge_asof(l, r, on="t", direction="nearest")
    return merged.set_index("index").sort_index()


def auto_trim_end(records, margin_sec):
    ends = []
    for key in ("cmd_hover", "cmd_full_state"):
        if records[key]:
            ends.append(max(float(r["t"]) for r in records[key]))
    if records["planned_paths"]:
        ends.append(max(float(p["t"]) for p in records["planned_paths"]))
    if not ends:
        return None
    return max(ends) + margin_sec


def load_solver_csv(run_dir, mode):
    if mode == "off":
        return None, None
    solves_path = os.path.join(run_dir, "all_solves.csv")
    events_path = os.path.join(run_dir, "solver_events.csv")
    if mode == "required" and not os.path.exists(solves_path):
        raise FileNotFoundError(solves_path)
    solves = read_numeric_csv(solves_path) if os.path.exists(solves_path) else None
    events = read_numeric_csv(events_path) if os.path.exists(events_path) else None
    return solves, events


def read_numeric_csv(path):
    df = pd.read_csv(path)
    if "solve_num" in df.columns:
        df = df[df["solve_num"].astype(str) != "solve_num"].copy()
    for col in df.columns:
        if col not in ("coord_mode", "event", "reason", "x0_source", "extra_values"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def position_cols(df):
    candidates = [
        ("x", "y", "z"),
        ("px", "py", "pz"),
        ("px_tgt", "py_tgt", "pz_tgt"),
        ("p0", "p1", "p2"),
        ("drone_x", "drone_y", "drone_z"),
    ]
    for cols in candidates:
        if df is not None and all(c in df.columns for c in cols):
            return list(cols)
    return []


def velocity_cols(df):
    candidates = [
        ("vx", "vy", "vz"),
        ("vx_tgt", "vy_tgt", "vz_tgt"),
        ("v0", "v1", "v2"),
    ]
    for cols in candidates:
        if df is not None and all(c in df.columns for c in cols):
            return list(cols)
    return []


def page_summary(title, topic_counts, drone, target, target_aligned, solves, events):
    fig = _fig(f"{title} — Summary", figsize=PAGE_TALL)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    lines = []
    lines.append("Bag/world truth")
    if not drone.empty and not target_aligned.empty:
        p_d = drone[["x", "y", "z"]].to_numpy(dtype=float)
        p_t = target_aligned[["tgt_x", "tgt_y", "tgt_z"]].to_numpy(dtype=float)
        valid = np.isfinite(p_d).all(axis=1) & np.isfinite(p_t).all(axis=1)
        if np.any(valid):
            err = np.linalg.norm(p_d[valid] - p_t[valid], axis=1)
            z_err = p_d[valid, 2] - p_t[valid, 2]
            lines.append(f"  final drone-target distance: {err[-1]:.4f} m")
            lines.append(f"  minimum drone-target distance: {np.min(err):.4f} m")
            lines.append(f"  final z error: {z_err[-1]:.4f} m")
            lines.append(f"  duration: {drone['time'].iloc[-1] - drone['time'].iloc[0]:.3f} s")

    if solves is not None and not solves.empty:
        meta = solves.groupby("solve_num", dropna=True).first().reset_index()
        if "solve_time_ms" in meta.columns:
            lines.append("")
            lines.append("Solver CSV")
            lines.append(f"  accepted solves: {len(meta)}")
            lines.append(f"  mean solve time: {meta['solve_time_ms'].mean():.2f} ms")
        pc = position_cols(solves)
        if pc:
            last = solves.sort_values(["solve_num", "node"]).groupby("solve_num").tail(1).tail(1)
            if not last.empty:
                terminal = last[pc].to_numpy(dtype=float)[0]
                lines.append(f"  final raw OCP terminal |p|: {np.linalg.norm(terminal):.4f}")

    if events is not None and not events.empty and "event" in events.columns:
        lines.append("")
        lines.append("Solver events")
        for event, count in events["event"].value_counts().items():
            lines.append(f"  {event}: {count}")

    lines.append("")
    lines.append("Topic counts")
    for topic, count in sorted(topic_counts.items()):
        if count:
            lines.append(f"  {topic}: {count}")

    ax.text(0.03, 0.95, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=11, color=TEXT_COL)
    return fig


def _plot_faded_path(ax, pts, color, label, speeds=None, cmap_name="viridis", lw=2.4):
    if len(pts) < 2:
        return
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    alpha = np.linspace(0.18, 0.95, len(segments))
    if speeds is not None and len(speeds) >= len(pts):
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        values = 0.5 * (speeds[:-1] + speeds[1:])
        norm = colors.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)) + 1e-9)
        base = cm.get_cmap(cmap_name)(norm(values))
        base[:, 3] = alpha
    else:
        import matplotlib.colors as colors
        rgba = np.asarray(colors.to_rgba(color))
        base = np.tile(rgba, (len(segments), 1))
        base[:, 3] = alpha
    coll = Line3DCollection(segments, colors=base, linewidths=lw)
    ax.add_collection3d(coll)
    ax.plot([], [], [], color=color, lw=lw, label=label)


def _scatter_time_samples(ax, pts, color, every=12):
    if len(pts) == 0:
        return
    idx = np.unique(np.concatenate([
        np.arange(0, len(pts), max(1, every)),
        np.array([len(pts) - 1]),
    ]))
    alpha = np.linspace(0.25, 1.0, len(idx))
    sizes = np.linspace(12, 42, len(idx))
    for a, s, i in zip(alpha, sizes, idx):
        ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2], color=color, alpha=float(a), s=float(s))


def _draw_body_axes(ax, drone, length=0.16, count=10):
    quat_cols = ["qw", "qx", "qy", "qz"]
    if not all(c in drone.columns for c in quat_cols) or drone.empty:
        return
    idx = np.linspace(0, len(drone) - 1, min(count, len(drone)), dtype=int)
    for i in idx:
        p = drone.iloc[i][["x", "y", "z"]].to_numpy(dtype=float)
        q = drone.iloc[i][quat_cols].to_numpy(dtype=float)
        R = quat_to_rotmat(q[0], q[1], q[2], q[3])
        age_alpha = 0.25 + 0.65 * (i / max(1, len(drone) - 1))
        ax.quiver(*p, *(R[:, 0] * length), color="#e84040", alpha=age_alpha, linewidth=1.0)
        ax.quiver(*p, *(R[:, 1] * length), color="#38c870", alpha=age_alpha, linewidth=1.0)
        ax.quiver(*p, *(R[:, 2] * length), color="#3898e8", alpha=age_alpha, linewidth=1.0)


def _set_equalish_limits(ax, drone_pts, target_pts):
    pts = np.vstack([drone_pts, target_pts])
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    centers = 0.5 * (mins + maxs)
    span = max(float(np.nanmax(maxs - mins)), 1.0)
    half = 0.55 * span
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(max(0.0, centers[2] - half), centers[2] + half)


def page_trajectory(title, drone, target, drone_odom):
    fig = _fig(f"{title} — World Trajectory", figsize=PAGE_GRID)
    drone_pts = drone[["x", "y", "z"]].to_numpy(dtype=float)
    target_pts = target[["tgt_x", "tgt_y", "tgt_z"]].to_numpy(dtype=float)
    speeds = None
    if not drone_odom.empty and all(c in drone_odom.columns for c in ("vx", "vy", "vz")):
        aligned_v = align_to(drone, drone_odom, ["vx", "vy", "vz"])
        speeds = np.linalg.norm(aligned_v[["vx", "vy", "vz"]].to_numpy(dtype=float), axis=1)

    for idx, az in enumerate(AZ_VIEWS):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        _style_3d(ax, f"World frame | azimuth {az} deg")
        ax.view_init(elev=22, azim=az)
        ax.set_box_aspect((1.0, 1.0, 0.65))
        _set_equalish_limits(ax, drone_pts, target_pts)
        _plot_faded_path(ax, drone_pts, C_DRONE, "Drone truth", speeds=speeds, cmap_name="plasma", lw=2.6)
        _plot_faded_path(ax, target_pts, C_TARGET, "Target truth", speeds=None, lw=2.2)
        _scatter_time_samples(ax, drone_pts, C_DRONE, every=max(1, len(drone_pts) // 18))
        _scatter_time_samples(ax, target_pts, C_TARGET, every=max(1, len(target_pts) // 18))
        _draw_body_axes(ax, drone)
        ax.scatter(drone_pts[0, 0], drone_pts[0, 1], drone_pts[0, 2], color=C_DRONE, marker="o", s=45)
        ax.scatter(drone_pts[-1, 0], drone_pts[-1, 1], drone_pts[-1, 2], color=C_DRONE, marker="*", s=85)
        ax.scatter(target_pts[-1, 0], target_pts[-1, 1], target_pts[-1, 2], color=C_TARGET, marker="*", s=75)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_components(title, actual, target, cols, labels, units, page_name):
    fig = _fig(f"{title} — {page_name}", figsize=PAGE_WIDE)
    ax = fig.add_subplot(1, 1, 1)
    _style_ax(ax, "Time [s]", units, page_name)
    for i, (ac, tc, label) in enumerate(zip(cols[0], cols[1], labels)):
        ax.plot(actual["time"].to_numpy(), actual[ac].to_numpy(),
                color=COMP_COLS[i], lw=2.0, label=f"drone {label}")
        ax.plot(actual["time"].to_numpy(), target[tc].to_numpy(),
                color=C_TARGET, lw=1.6, ls="--", label=f"target {label}")
    ax.legend(loc="best", ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_error(title, drone, target_aligned):
    fig = _fig(f"{title} — Drone-Target Error", figsize=PAGE_WIDE)
    p_d = drone[["x", "y", "z"]].to_numpy(dtype=float)
    p_t = target_aligned[["tgt_x", "tgt_y", "tgt_z"]].to_numpy(dtype=float)
    err = p_d - p_t
    dist = np.linalg.norm(err, axis=1)
    labels = ["dx", "dy", "dz"]
    ax = fig.add_subplot(1, 1, 1)
    _style_ax(ax, "Time [s]", "m", "Drone - target world error")
    for i in range(3):
        ax.plot(drone["time"].to_numpy(), err[:, i], color=COMP_COLS[i], lw=1.8, label=labels[i])
    ax.plot(drone["time"].to_numpy(), dist, color=C_ERR, lw=2.2, label="distance norm")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.legend(loc="best", ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_hover(title, drone_odom, hover):
    if hover.empty:
        return None
    aligned = align_to(hover, drone_odom, ["vx", "vy", "vz", "z"])
    fig = _fig(f"{title} — cmd_hover", figsize=PAGE_WIDE)
    pairs = [
        ("vx", "vx", "vx [m/s]"),
        ("vy", "vy", "vy [m/s]"),
        ("z_distance", "z", "z [m]"),
        ("yaw_rate", None, "yaw rate [rad/s or deg/s]"),
    ]
    ax = fig.add_subplot(1, 1, 1)
    _style_ax(ax, "Time [s]", "cmd_hover value", "cmd_hover and physical response")
    for i, (cmd_col, act_col, label) in enumerate(pairs):
        ax.plot(hover["time"].to_numpy(), hover[cmd_col].to_numpy(),
                color=COMP_COLS[i % len(COMP_COLS)], lw=1.8, label=f"cmd {cmd_col}")
        if act_col and act_col in aligned.columns:
            ax.plot(hover["time"].to_numpy(), aligned[act_col].to_numpy(),
                    color=COMP_COLS[i % len(COMP_COLS)], lw=1.4, ls="--", label=f"actual {act_col}")
    ax.legend(loc="best", ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_fullstate(title, drone_odom, cmd):
    if cmd.empty:
        return None
    compare_cols = ["x", "y", "z", "vx", "vy", "vz"]
    aligned = align_to(cmd, drone_odom, compare_cols)
    fig = _fig(f"{title} — cmd_full_state", figsize=PAGE_TALL)
    for i, col in enumerate(compare_cols):
        ax = fig.add_subplot(3, 2, i + 1)
        _style_ax(ax, "Time [s]", col, col)
        ax.plot(cmd["time"].to_numpy(), cmd[col].to_numpy(),
                color=C_CMD, lw=1.7, label=f"cmd {col}")
        if col in aligned.columns:
            ax.plot(cmd["time"].to_numpy(), aligned[col].to_numpy(),
                    color=C_DRONE, lw=1.4, ls="--", label=f"actual {col}")
        ax.legend(loc="best", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_target_frame(title, tf_df):
    if tf_df.empty:
        return None
    fig = _fig(f"{title} — Target-frame Odometry Diagnostic", figsize=PAGE_TALL)
    groups = [
        (["tf_x", "tf_y", "tf_z"], "p_B^N [m]"),
        (["tf_vx", "tf_vy", "tf_vz"], "v_B^N [m/s]"),
        (["tf_qw", "tf_qx", "tf_qy", "tf_qz"], "q_NB"),
    ]
    for i, (cols, label) in enumerate(groups):
        ax = fig.add_subplot(3, 1, i + 1)
        _style_ax(ax, "Time [s]", label, label)
        for j, col in enumerate(cols):
            if col in tf_df.columns:
                ax.plot(tf_df["time"].to_numpy(), tf_df[col].to_numpy(),
                        color=COMP_COLS[j % len(COMP_COLS)], lw=1.8, label=col)
        ax.legend(loc="best", ncol=len(cols))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def page_solver_stats(title, solves, events):
    if solves is None or solves.empty:
        return None
    meta = solves.groupby("solve_num", dropna=True).first().reset_index()
    fig = _fig(f"{title} — Solver Timing", figsize=PAGE_WIDE)
    ax = fig.add_subplot(1, 1, 1)
    _style_ax(ax, "Solve #", "ms", "Accepted solve time and iterations")
    ax.bar(meta["solve_num"].to_numpy(), meta.get("solve_time_ms", np.nan).to_numpy(),
           color="#8b9cae", alpha=0.65, label="solve time [ms]")
    ax2 = ax.twinx()
    ax2.plot(meta["solve_num"].to_numpy(), meta.get("solve_iters", np.nan).to_numpy(),
             color="#9467bd", marker="o", label="iterations")
    ax2.set_ylabel("iterations")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    if events is not None and not events.empty and "event" in events.columns:
        subtitle = ", ".join(f"{k}={v}" for k, v in events["event"].value_counts().items())
        ax.text(0.01, 0.95, f"events: {subtitle}", transform=ax.transAxes, va="top")
    return fig


def page_ocp_nodes(title, solves):
    if solves is None or solves.empty:
        return None
    pc = position_cols(solves)
    vc = velocity_cols(solves)
    if not pc and not vc:
        return None
    fig = _fig(f"{title} — OCP-frame Solver Nodes", figsize=PAGE_TALL)
    coord = str(solves["coord_mode"].dropna().iloc[0]) if "coord_mode" in solves.columns and solves["coord_mode"].notna().any() else "unknown"
    groups = []
    if pc:
        groups.append((pc, "raw OCP position"))
    if vc:
        groups.append((vc, "raw OCP velocity"))
    for i, (cols, label) in enumerate(groups):
        ax = fig.add_subplot(len(groups), 1, i + 1)
        _style_ax(ax, "Node time [s]", label, f"{label} ({coord})")
        for solve_num, grp in solves.groupby("solve_num"):
            grp = grp.sort_values("node")
            alpha = 0.20
            lw = 1.0
            if solve_num == solves["solve_num"].max():
                alpha = 0.95
                lw = 2.0
            for j, col in enumerate(cols):
                ax.plot(grp["t"].to_numpy(), grp[col].to_numpy(),
                        color=COMP_COLS[j], alpha=alpha, lw=lw)
        ax.legend([Line2D([0], [0], color=COMP_COLS[j], lw=2) for j in range(len(cols))], cols)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def build_pdf(run_dir, bag_dir, out_path, cut_secs, csv_solver, auto_trim, trim_margin_sec):
    records, topic_counts, _ = read_bag(bag_dir)
    drone_pose = df_from(records["drone_pose"])
    drone_odom = df_from(records["drone_odom"])
    target = df_from(records["target_odom"])
    tf_odom = df_from(records["target_frame_odom"])
    hover = df_from(records["cmd_hover"])
    fullstate = df_from(records["cmd_full_state"])
    paths = records["planned_paths"]

    drone = drone_pose if not drone_pose.empty else drone_odom
    if drone.empty:
        raise RuntimeError("No drone truth topic found. Expected /cf_1/pose or /cf_1/odom.")
    if target.empty:
        raise RuntimeError("No target truth topic found. Expected /target/odom.")
    if drone_odom.empty:
        drone_odom = drone.copy()

    trim_end_abs = None
    if auto_trim and cut_secs is None:
        trim_end_abs = auto_trim_end(records, trim_margin_sec)

    t0_abs = normalize_times([drone, drone_odom, target, tf_odom, hover, fullstate], paths)
    if trim_end_abs is not None:
        cut_secs = max(0.0, trim_end_abs - t0_abs)

    if cut_secs is not None:
        for df in (drone, drone_odom, target, tf_odom, hover, fullstate):
            if not df.empty:
                df.drop(df[df["time"] > cut_secs].index, inplace=True)
        paths = [p for p in paths if p["time"] <= cut_secs]

    target_aligned = align_to(drone, target, ["tgt_x", "tgt_y", "tgt_z", "tgt_vx", "tgt_vy", "tgt_vz"])
    target_for_vel = align_to(drone_odom, target, ["tgt_vx", "tgt_vy", "tgt_vz"])
    solves, events = load_solver_csv(run_dir, csv_solver)

    title = os.path.basename(os.path.normpath(run_dir))
    if out_path is None:
        out_path = os.path.join(run_dir, f"{title}_bag_analysis.pdf")

    with PdfPages(out_path) as pdf:
        pages = [
            page_summary(title, topic_counts, drone, target, target_aligned, solves, events),
            page_trajectory(title, drone, target, drone_odom),
            page_components(title, drone, target_aligned,
                            (["x", "y", "z"], ["tgt_x", "tgt_y", "tgt_z"]),
                            ["x", "y", "z"], "m", "World Position"),
            page_error(title, drone, target_aligned),
        ]
        if all(c in drone_odom.columns for c in ("vx", "vy", "vz")) and not target_for_vel.empty:
            pages.append(page_components(title, drone_odom, target_for_vel,
                                         (["vx", "vy", "vz"], ["tgt_vx", "tgt_vy", "tgt_vz"]),
                                         ["vx", "vy", "vz"], "m/s", "World Velocity"))
        pages.extend([
            page_hover(title, drone_odom, hover),
            page_fullstate(title, drone_odom, fullstate),
            page_target_frame(title, tf_odom),
            page_solver_stats(title, solves, events),
            page_ocp_nodes(title, solves),
        ])
        for fig in pages:
            if fig is not None:
                pdf.savefig(fig, facecolor=FIG_BG)
                plt.close(fig)

        info = pdf.infodict()
        info["Title"] = f"CoManDO bag analysis: {title}"
        info["Subject"] = "Bag-first trajectory and solver analysis"

    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate bag-first CoManDO circle/landing analysis PDF")
    parser.add_argument("--dir", required=True, help="Run log directory")
    parser.add_argument("--bag", default=None, help="Bag directory, default: <dir>/bags/comando_debug")
    parser.add_argument("--out", default=None, help="Output PDF path")
    parser.add_argument("--cut", type=float, default=None, help="Keep only first N seconds")
    parser.add_argument("--no-auto-trim", action="store_true",
                        help="Disable default trimming to active command/path window")
    parser.add_argument("--trim-margin", type=float, default=2.0,
                        help="Seconds kept after last command/path when auto trimming")
    parser.add_argument("--csv-solver", choices=("auto", "off", "required"), default="auto",
                        help="Use all_solves.csv/solver_events.csv for solver pages")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.dir)
    bag_dir = os.path.abspath(args.bag) if args.bag else os.path.join(run_dir, "bags", "comando_debug")
    build_pdf(run_dir, bag_dir, args.out, args.cut, args.csv_solver,
              auto_trim=not args.no_auto_trim,
              trim_margin_sec=args.trim_margin)


if __name__ == "__main__":
    main()
