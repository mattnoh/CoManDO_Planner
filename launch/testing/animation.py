#!/usr/bin/env python3
"""
manim_receding_horizon_animation.py
====================================
Manim Community Edition — 3D receding-horizon MPC replanning animation.

Scene: RecedingHorizonMPC
  • Drone (small sphere + RGB body-frame axes)
  • Actual executed path traced as a solid line
  • Each MPC solve draws its planned trajectory as a dotted path
  • Ghost fading: current = full, 1 old = medium, 2 old = very faint, 3+ removed
  • First-solve trajectory remains permanently visible as a reference
  • Solve counter + solve-time text in the corner

CSV layout (planner_node.cpp convention):
  all_solves.csv        solve_num, solve_time_ms, node,
                        t, x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz
  actual_state.csv      timestamp, x,y,z, vx,vy,vz,
                        qw,qx,qy,qz, wx,wy,wz

Usage — from the directory containing the CSV files:
  manim -pqh manim_receding_horizon_animation.py RecedingHorizonMPC

Or specify a folder explicitly via environment variable:
  LOG_FOLDER=/path/to/logs manim -pqh manim_receding_horizon_animation.py RecedingHorizonMPC

Rendering flags (common):
  -ql  low quality (480p, fast preview)
  -qm  medium quality (720p)
  -qh  high quality (1080p)
  -p   preview on completion
  --save_as_gif   export GIF instead of MP4
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Manim import guard ─────────────────────────────────────────────────────────
try:
    from manim import (
        ThreeDScene, ThreeDAxes,
        Sphere, Arrow3D, Line3D, DashedVMobject, VMobject,
        Dot3D, MathTex, Text, VGroup, DecimalNumber,
        ORIGIN, UP, DOWN, LEFT, RIGHT, OUT, IN, UL, UR, DL, DR,
        RED, GREEN, BLUE, WHITE, YELLOW, ORANGE, PURPLE, GRAY,
        DEGREES,
        Create, Write, FadeIn, FadeOut, Transform,
        AnimationGroup, Succession, LaggedStart,
        rate_functions, config,
        ManimColor,
    )
    from manim import MovingCameraScene  # noqa: F401 — unused but available
    from manim import logger as manim_logger
except ImportError as exc:
    sys.exit(
        f"Manim not installed: {exc}\n"
        "Install with:  pip install manim"
    )

# ── Numeric types for Manim colour specs ──────────────────────────────────────
try:
    from manim import color_to_rgba  # type: ignore
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Folder containing the CSV logs.  Override via LOG_FOLDER env variable.
LOG_FOLDER = os.environ.get("LOG_FOLDER", ".")

# Down-sample all_solves so the animation stays under ~5 minutes.
# Set to 1 to use every solve.
SOLVE_SUBSAMPLE = 1          # use every Nth solve (1 = all)

# Time (seconds) to draw each planned trajectory
DRAW_TIME = 1.2              # seconds for the dotted-line draw animation

# Time (seconds) the drone moves between two replanning events
FLY_TIME  = 0.6              # seconds of flight per solve interval

# Number of actual-state steps to show per solve interval
ACT_STEPS_PER_SOLVE = 10

# Ghost opacity schedule: index = age (0 = current solve)
GHOST_ALPHA = {0: 1.0, 1: 0.45, 2: 0.14}   # 3+ → removed

# Scale for converting metres to Manim units (1 m → 1 Manim unit by default)
SCALE = 1.0

# Body-frame axis arrow length (Manim units)
BF_LEN = 0.22

# ── Colour constants (Manim hex strings) ──────────────────────────────────────
COL_FIRST  = "#D97706"   # first-solve plan (amber)
COL_GHOST  = "#818CF8"   # ghost plans (indigo)
COL_ACTUAL = "#EF4444"   # actual executed path (red)
COL_DRONE  = "#0891B2"   # drone body (cyan)
COL_TEXT   = "#111827"   # info text
COL_BG     = "#F8FAFC"   # scene background


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

SOLVE_COLS = [
    "solve_num", "solve_time_ms", "node",
    "t", "x", "y", "z",
    "vx", "vy", "vz",
    "qw", "qx", "qy", "qz",
    "wx", "wy", "wz",
]
ACT_COLS = [
    "timestamp", "x", "y", "z",
    "vx", "vy", "vz",
    "qw", "qx", "qy", "qz",
    "wx", "wy", "wz",
]


def load_all_solves(folder: str) -> pd.DataFrame:
    path = Path(folder) / "all_solves.csv"
    if not path.exists():
        raise FileNotFoundError(f"all_solves.csv not found in {folder}")
    raw = pd.read_csv(path, header=0)
    n = min(len(SOLVE_COLS), raw.shape[1])
    raw = raw.iloc[:, :n].copy()
    raw.columns = SOLVE_COLS[:n]
    raw["solve_num"] = raw["solve_num"].astype(int)
    raw["node"]      = raw["node"].astype(int)
    return raw


def load_actual_states(folder: str) -> pd.DataFrame | None:
    path = Path(folder) / "actual_state.csv"
    if not path.exists():
        manim_logger.warning("actual_state.csv not found — drone will follow solve node 0")
        return None
    raw = pd.read_csv(path, header=0)
    n = min(len(ACT_COLS), raw.shape[1])
    raw = raw.iloc[:, :n].copy()
    raw.columns = ACT_COLS[:n]
    return raw


def extract_solve_trajectories(df: pd.DataFrame, subsample: int = 1):
    """
    Returns a list of (solve_num, solve_time_ms, xyz_array) tuples.
    xyz_array shape: (n_nodes, 3), ordered by node index.
    """
    solve_nums = sorted(df["solve_num"].unique())
    if subsample > 1:
        solve_nums = solve_nums[::subsample]

    result = []
    for sn in solve_nums:
        grp = df[df["solve_num"] == sn].sort_values("node")
        tms = grp["solve_time_ms"].iloc[0]
        xyz = grp[["x", "y", "z"]].values * SCALE
        result.append((int(sn), float(tms), xyz))
    return result


def extract_drone_positions(solves_raw: pd.DataFrame) -> np.ndarray:
    """One position per solve: the node==0 state."""
    rows = []
    for sn, grp in solves_raw.groupby("solve_num", sort=True):
        node0 = grp[grp["node"] == 0]
        if len(node0):
            rows.append(node0.iloc[0][["x", "y", "z"]].values * SCALE)
    return np.array(rows) if rows else np.zeros((1, 3))


def extract_drone_quats(solves_raw: pd.DataFrame) -> np.ndarray:
    """One quaternion per solve: the node==0 quaternion."""
    rows = []
    for sn, grp in solves_raw.groupby("solve_num", sort=True):
        node0 = grp[grp["node"] == 0]
        if len(node0):
            rows.append(node0.iloc[0][["qw", "qx", "qy", "qz"]].values)
    return np.array(rows) if rows else np.tile([1, 0, 0, 0], (1, 1))


def quat_to_rotation_matrix(qw, qx, qy, qz) -> np.ndarray:
    """Unit quaternion → 3×3 rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q) + 1e-12
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z),  2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),    1 - 2*(x*x+z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),    2*(y*z + w*x),   1 - 2*(x*x+y*y)],
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  Manim helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_drone(pos: np.ndarray, R: np.ndarray, bf_len: float = BF_LEN) -> VGroup:
    """
    Drone model: small sphere body + 3 body-frame axis arrows.
      X axis → red
      Y axis → green
      Z axis → blue
    """
    body = Sphere(radius=0.09, color=COL_DRONE)
    body.move_to(pos)

    axes = VGroup()
    for ci, col in enumerate([RED, GREEN, BLUE]):
        direction = R[:, ci] * bf_len
        arrow = Arrow3D(
            start=pos,
            end=pos + direction,
            color=col,
            thickness=0.025,
            base_radius=0.018,
            height=0.06,
        )
        axes.add(arrow)

    return VGroup(body, axes)


def make_plan_path(xyz: np.ndarray, color: str, stroke_width: float = 3.0,
                   dashed_ratio: float = 0.5) -> DashedVMobject:
    """Build a dotted/dashed 3-D polyline from an Nx3 position array.
    DashedVMobject in Manim 0.19 accepts num_dashes and dashed_ratio only —
    dash_length was removed."""
    path = VMobject()
    path.set_points_as_corners([np.array([x, y, z]) for x, y, z in xyz])
    path.set_stroke(color=color, width=stroke_width)
    dashed = DashedVMobject(path, num_dashes=max(8, len(xyz) * 3),
                            dashed_ratio=dashed_ratio)
    return dashed


def make_actual_path(points: list[np.ndarray], color: str = COL_ACTUAL,
                     stroke_width: float = 2.5) -> VMobject:
    """Solid line tracing the executed path."""
    path = VMobject()
    if len(points) < 2:
        path.set_points_as_corners([np.zeros(3), np.zeros(3)])
    else:
        path.set_points_as_corners(points)
    path.set_stroke(color=color, width=stroke_width)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Scene
# ═══════════════════════════════════════════════════════════════════════════════

class RecedingHorizonMPC(ThreeDScene):
    """
    Main Manim scene.

    Timeline per solve k (k > 0):
      1. Text update: "Solve k — Solve Time: XX ms"
      2. Draw new planned trajectory (dotted), 1–2 s
      3. Fade ghosts (age-based transparency)
      4. Drone moves along actual trajectory to next solve position, FLY_TIME s
      5. Repeat

    Solve 0 (initial):
      1. Show text "Solve 0 — Solve Time: XX ms"
      2. Draw first planned trajectory (dotted)
      3. Drone placed at initial position
    """

    def construct(self):
        # ── Background ──────────────────────────────────────────────────────────
        self.camera.background_color = COL_BG

        # ── Load data ───────────────────────────────────────────────────────────
        manim_logger.info(f"Loading logs from: {LOG_FOLDER}")
        try:
            solves_raw = load_all_solves(LOG_FOLDER)
        except FileNotFoundError as exc:
            manim_logger.error(str(exc))
            self._show_error(str(exc))
            return

        actual_df = load_actual_states(LOG_FOLDER)

        # Solve trajectories: list of (solve_num, solve_time_ms, xyz_array)
        solve_list = extract_solve_trajectories(solves_raw, SOLVE_SUBSAMPLE)
        n_solves   = len(solve_list)

        # Per-solve drone positions and orientations (from node 0)
        drone_positions = extract_drone_positions(solves_raw)
        drone_quats     = extract_drone_quats(solves_raw)

        # Sub-sample drone_positions / quats to match solve_list length
        all_solve_nums = sorted(solves_raw["solve_num"].unique())
        if SOLVE_SUBSAMPLE > 1:
            used_nums = all_solve_nums[::SOLVE_SUBSAMPLE]
            idx_map   = {sn: i for i, sn in enumerate(all_solve_nums)}
            keep_idx  = [idx_map[sn] for sn in used_nums if sn in idx_map]
            drone_positions = drone_positions[keep_idx]
            drone_quats     = drone_quats[keep_idx]

        drone_positions = drone_positions[:n_solves]
        drone_quats     = drone_quats[:n_solves]

        # Actual state positions (evenly sub-sampled to ACT_STEPS_PER_SOLVE per solve)
        if actual_df is not None:
            act_xyz = actual_df[["x", "y", "z"]].values * SCALE
        else:
            # Fall back to drone_positions as the "actual" path
            act_xyz = drone_positions.copy()

        # Compute axis extent for camera placement
        all_xyz = np.vstack([s[2] for s in solve_list])
        center   = all_xyz.mean(axis=0)
        extent   = max(
            float(all_xyz[:, 0].max() - all_xyz[:, 0].min()),
            float(all_xyz[:, 1].max() - all_xyz[:, 1].min()),
            float(all_xyz[:, 2].max() - all_xyz[:, 2].min()),
            0.5,
        )

        # ── Camera ──────────────────────────────────────────────────────────────
        # Phi = angle from top (0 = top-down), Theta = azimuth
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES,
                                    zoom=0.7)
        self.camera.frame_center = center

        # ── 3-D axes (optional, kept small) ────────────────────────────────────
        ax_len = extent * 0.18
        axes = ThreeDAxes(
            x_range=[-ax_len, ax_len, ax_len / 2],
            y_range=[-ax_len, ax_len, ax_len / 2],
            z_range=[0, ax_len * 2,  ax_len / 2],
            x_length=ax_len * 2, y_length=ax_len * 2, z_length=ax_len * 2,
            tips=False,
        )
        axes.shift(center - axes.get_center())

        # ── Corner info text ────────────────────────────────────────────────────
        # Fixed-frame text (2-D overlay; ThreeDScene keeps 2-D mobjects as overlays)
        info_text = Text("Solve 0 — Solve Time: -- ms",
                         font_size=24, color=COL_TEXT)
        info_text.to_corner(UL, buff=0.25)
        self.add_fixed_in_frame_mobjects(info_text)

        # ── Actual path (drawn incrementally) ───────────────────────────────────
        # We'll accumulate actual path points and redraw each step.
        # Start with the first position.
        actual_trace_points: list[np.ndarray] = [drone_positions[0].copy()]
        actual_line = make_actual_path(actual_trace_points)
        self.add(actual_line)

        # ── Drone model ─────────────────────────────────────────────────────────
        R0 = quat_to_rotation_matrix(*drone_quats[0])
        drone_mob = make_drone(drone_positions[0], R0)
        self.add(drone_mob)

        # ── Add axes ────────────────────────────────────────────────────────────
        self.add(axes)

        # ── Ghost plan storage: deque of (solve_num, DashedVMobject) ────────────
        # Index 0 = most recent (current).
        active_plans: list[tuple[int, DashedVMobject]] = []

        # ── First-solve trajectory — permanent reference ─────────────────────────
        sn0, tms0, xyz0 = solve_list[0]
        first_plan = make_plan_path(xyz0, color=COL_FIRST, stroke_width=4.5,
                                    dashed_ratio=0.6)
        first_plan.set_opacity(0.90)

        # Update info text for solve 0
        new_info = Text(f"Solve {sn0} — Solve Time: {tms0:.1f} ms",
                        font_size=24, color=COL_TEXT)
        new_info.to_corner(UL, buff=0.25)
        self.add_fixed_in_frame_mobjects(new_info)
        self.remove(info_text)
        info_text = new_info

        # Animate drawing of the first plan
        self.begin_ambient_camera_rotation(rate=0.04)
        self.play(Create(first_plan), run_time=DRAW_TIME)
        self.add(first_plan)

        # Track first_plan separately (it never fades)

        # ── Main replanning loop ─────────────────────────────────────────────────
        for step_i in range(n_solves):
            sn, tms, xyz = solve_list[step_i]

            # ── 1. Update info text ────────────────────────────────────────────
            new_info = Text(f"Solve {sn} — Solve Time: {tms:.1f} ms",
                            font_size=24, color=COL_TEXT)
            new_info.to_corner(UL, buff=0.25)
            self.add_fixed_in_frame_mobjects(new_info)
            self.remove(info_text)
            info_text = new_info

            # ── 2. Build new planned trajectory ───────────────────────────────
            if step_i == 0:
                # Solve 0 already drawn as first_plan; add it to active_plans
                current_plan = make_plan_path(xyz, color=COL_GHOST,
                                              stroke_width=3.2)
                current_plan.set_opacity(0.0)   # hidden (first_plan is visible)
                active_plans.insert(0, (sn, current_plan))
                self.add(current_plan)
            else:
                current_plan = make_plan_path(xyz, color=COL_GHOST,
                                              stroke_width=3.2)
                current_plan.set_opacity(0.0)

                # ── 3. Fade ghosts according to age schedule ──────────────────
                anims = [Create(current_plan)]  # draw new plan
                for age, (_, ghost) in enumerate(active_plans):
                    if age + 1 in GHOST_ALPHA:
                        target_alpha = GHOST_ALPHA[age + 1]
                        # Manim FadeOut / set_opacity via Transform
                        ghost_copy = ghost.copy().set_opacity(target_alpha)
                        anims.append(Transform(ghost, ghost_copy))
                    else:
                        # 3+ old → fade out completely and remove
                        anims.append(FadeOut(ghost))

                # Remove plans that are 3+ old from the active list
                active_plans = [(s, m) for i, (s, m) in enumerate(active_plans)
                                if i + 1 in GHOST_ALPHA]

                current_plan.set_opacity(GHOST_ALPHA[0])
                active_plans.insert(0, (sn, current_plan))
                self.add(current_plan)

                self.play(AnimationGroup(*anims, lag_ratio=0.0),
                          run_time=DRAW_TIME)

            # ── 4. Move drone along actual trajectory to next position ─────────
            if step_i < n_solves - 1:
                next_pos  = drone_positions[step_i + 1]
                next_quat = drone_quats[step_i + 1]
                R_next    = quat_to_rotation_matrix(*next_quat)

                # Compute actual path segment for this interval
                n_act = len(act_xyz)
                seg_start = int(step_i / n_solves * n_act)
                seg_end   = int((step_i + 1) / n_solves * n_act)
                seg_end   = min(seg_end, n_act)
                seg_pts   = act_xyz[seg_start:seg_end]

                # Build new drone mobject at next position
                new_drone = make_drone(next_pos, R_next)

                # Extend actual trace
                actual_trace_points.extend(seg_pts.tolist())
                new_actual_line = make_actual_path(actual_trace_points)

                self.play(
                    Transform(drone_mob, new_drone),
                    Transform(actual_line, new_actual_line),
                    run_time=FLY_TIME,
                    rate_func=rate_functions.smooth,
                )

        # ── Hold final frame ─────────────────────────────────────────────────────
        self.stop_ambient_camera_rotation()
        self.wait(2.0)

    # ── Error display helper ─────────────────────────────────────────────────────

    def _show_error(self, message: str):
        err = Text(message, font_size=26, color=RED)
        self.add_fixed_in_frame_mobjects(err)
        self.wait(3)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point — not executed by Manim directly but useful for debugging
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(__doc__)
    print("\nTo render, run:")
    print("  manim -pqh manim_receding_horizon_animation.py RecedingHorizonMPC")
    print("  LOG_FOLDER=/path/to/logs manim -pqh manim_receding_horizon_animation.py RecedingHorizonMPC")