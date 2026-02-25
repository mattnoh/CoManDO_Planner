#!/usr/bin/env python3
"""
Plot logged flight data from CoManDO MPC planner

Usage:
    python3 logger.py /path/to/flight_folder [--cutoff TIME] [--all] [--latest]

CSV file structure
──────────────────
commanded_state.csv
    timestamp, x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz,
    fx,fy,fz, mx,my,mz, thrust_norm
    col:   0   1 2 3   4  5  6   7  8  9 10  11 12 13
          14 15 16  17 18 19       20

actual_state.csv
    timestamp, x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz
    col:   0   1 2 3   4  5  6   7  8  9 10  11 12 13

first_solve_trajectory.csv
    node, t, x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz,
    fx,fy,fz, mx,my,mz, thrust_norm
    col:  0  1  2 3 4   5  6  7   8  9 10 11  12 13 14
         15 16 17  18 19 20       21
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ── Column indices ─────────────────────────────────────────────────────────────
# commanded_state / actual_state share columns 0-13
C_TS  = 0
C_X,  C_Y,  C_Z  = 1,  2,  3
C_VX, C_VY, C_VZ = 4,  5,  6
C_QW, C_QX, C_QY, C_QZ = 7, 8, 9, 10
C_WX, C_WY, C_WZ = 11, 12, 13
# commanded_state only
C_FX, C_FY, C_FZ = 14, 15, 16
C_MX, C_MY, C_MZ = 17, 18, 19
C_TNORM           = 20

# first_solve_trajectory columns
F_NODE, F_T      = 0, 1
F_X,  F_Y,  F_Z  = 2,  3,  4
F_VX, F_VY, F_VZ = 5,  6,  7
F_QW, F_QX, F_QY, F_QZ = 8, 9, 10, 11
F_WX, F_WY, F_WZ = 12, 13, 14
F_FX, F_FY, F_FZ = 15, 16, 17
F_MX, F_MY, F_MZ = 18, 19, 20
F_TNORM           = 21


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_csv(filepath, cutoff_time=None):
    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data is None or data.ndim < 2 or data.shape[0] == 0:
            return None
        if cutoff_time is not None:
            rel  = data[:, 0] - data[0, 0]
            mask = rel <= cutoff_time
            if not np.any(mask):
                print(f"Warning: cutoff {cutoff_time}s is before all data in {filepath}")
                return None
            data = data[mask]
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_first_traj(filepath):
    """Load first_solve_trajectory.csv — no cutoff, uses horizon time column."""
    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data is None or data.ndim < 2 or data.shape[0] == 0:
            return None
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def quat_rotate(qwxyz, v):
    qw, qx, qy, qz = qwxyz
    vx, vy, vz = v
    t2, t3, t4  =  qw*qx,  qw*qy,  qw*qz
    t5, t6, t7  = -qx*qx,  qx*qy,  qx*qz
    t8, t9, t10 = -qy*qy,  qy*qz, -qz*qz
    return np.array([
        2*((t8+t10)*vx + (t6-t4)*vy + (t3+t7)*vz) + vx,
        2*((t4+t6)*vx + (t5+t10)*vy + (t9-t2)*vz) + vy,
        2*((t7-t3)*vx + (t2+t9)*vy + (t5+t8)*vz) + vz,
    ])


# ── Comprehensive time-series plot ────────────────────────────────────────────
def plot_flight_data(folder_path, cutoff_time=None):
    cmd = load_csv(os.path.join(folder_path, "commanded_state.csv"), cutoff_time)
    act = load_csv(os.path.join(folder_path, "actual_state.csv"),    cutoff_time)
    fs  = load_first_traj(os.path.join(folder_path, "first_solve_trajectory.csv"))

    if cmd is None or act is None:
        print("Error: could not load commanded_state.csv or actual_state.csv")
        return

    cmd_t = cmd[:, C_TS] - cmd[0, C_TS]
    act_t = act[:, C_TS] - act[0, C_TS]

    has_fs = fs is not None and fs.shape[1] > F_TNORM
    fs_t   = fs[:, F_T] if has_fs else None

    fig, axes = plt.subplots(4, 2, figsize=(17, 18))

    # ── Position ──────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for c, lbl, ac, fc in zip(['r','g','b'], ['X','Y','Z'],
                               [C_X, C_Y, C_Z], [F_X, F_Y, F_Z]):
        ax.plot(act_t, act[:, ac],  color=c, linestyle='-',  lw=2,        label=f'Actual {lbl}')
        ax.plot(cmd_t, cmd[:, ac],  color=c, linestyle='--', lw=1, alpha=0.7, label=f'Cmd {lbl}')
        if has_fs:
            ax.plot(fs_t, fs[:, fc], color=c, linestyle=':', lw=1.5, alpha=0.55, label=f'Init {lbl}')
    ax.set_ylabel('Position (m)')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.grid(True); ax.set_title('Position')

    # ── Velocity ──────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    for c, lbl, ac, fc in zip(['r','g','b'], ['Vx','Vy','Vz'],
                               [C_VX, C_VY, C_VZ], [F_VX, F_VY, F_VZ]):
        ax.plot(act_t, act[:, ac],  color=c, linestyle='-',  lw=2,        label=f'Actual {lbl}')
        ax.plot(cmd_t, cmd[:, ac],  color=c, linestyle='--', lw=1, alpha=0.7, label=f'Cmd {lbl}')
        if has_fs:
            ax.plot(fs_t, fs[:, fc], color=c, linestyle=':', lw=1.5, alpha=0.55, label=f'Init {lbl}')
    ax.set_ylabel('Velocity (m/s)')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.grid(True); ax.set_title('Velocity')

    # ── Quaternion ────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    # Use explicit color and linestyle because 'orange' is not a single-character code
    for c, lbl, ac, fc in zip(['r','g','b','orange'], ['qw','qx','qy','qz'],
                               [C_QW, C_QX, C_QY, C_QZ],
                               [F_QW, F_QX, F_QY, F_QZ]):
        ax.plot(act_t, act[:, ac],  color=c, linestyle='-',  lw=2,        label=f'Act {lbl}')
        ax.plot(cmd_t, cmd[:, ac],  color=c, linestyle='--', lw=1, alpha=0.7, label=f'Cmd {lbl}')
        if has_fs:
            ax.plot(fs_t, fs[:, fc], color=c, linestyle=':', lw=1.5, alpha=0.55, label=f'Init {lbl}')
    ax.set_ylabel('Quaternion')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.grid(True); ax.set_title('Orientation (Quaternions)')

    # ── Angular velocity ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    for c, lbl, ac, fc in zip(['r','g','b'], ['ωx','ωy','ωz'],
                               [C_WX, C_WY, C_WZ], [F_WX, F_WY, F_WZ]):
        ax.plot(act_t, act[:, ac],  color=c, linestyle='-',  lw=2,        label=f'Act {lbl}')
        ax.plot(cmd_t, cmd[:, ac],  color=c, linestyle='--', lw=1, alpha=0.7, label=f'Cmd {lbl}')
        if has_fs:
            ax.plot(fs_t, fs[:, fc], color=c, linestyle=':', lw=1.5, alpha=0.55, label=f'Init {lbl}')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.grid(True); ax.set_title('Angular Velocity')

    # ── Control forces ────────────────────────────────────────────────────────
    ax = axes[2, 0]
    for c, lbl, cc, fc in zip(['r','g','b'], ['Fx','Fy','Fz'],
                               [C_FX, C_FY, C_FZ], [F_FX, F_FY, F_FZ]):
        ax.plot(cmd_t, cmd[:, cc], color=c, linestyle='-',  lw=2,        label=f'Cmd {lbl}')
        if has_fs:
            ax.plot(fs_t, fs[:, fc], color=c, linestyle=':', lw=1.5, alpha=0.6, label=f'Init {lbl}')
    ax.set_ylabel('Force (N)'); ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True); ax.set_title('Control Forces')

    # ── Control moments + thrust ──────────────────────────────────────────────
    ax = axes[2, 1]
    for c, lbl, cc, fc in zip(['r','g','b'], ['Mx','My','Mz'],
                               [C_MX, C_MY, C_MZ], [F_MX, F_MY, F_MZ]):
        ax.plot(cmd_t, cmd[:, cc], color=c, linestyle='-',  lw=2,        label=f'Cmd {lbl}')
        if has_fs:
            ax.plot(fs_t, fs[:, fc], color=c, linestyle=':', lw=1.5, alpha=0.6, label=f'Init {lbl}')
    ax.set_ylabel('Moment (Nm)'); ax.set_xlabel('Time (s)')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.grid(True)

    ax2b = ax.twinx()
    ax2b.plot(cmd_t, cmd[:, C_TNORM], 'k-',  lw=2, alpha=0.85, label='Thrust (cmd)')
    if has_fs:
        ax2b.plot(fs_t, fs[:, F_TNORM], 'k:', lw=1.5, alpha=0.55, label='Thrust (init)')
    ax2b.set_ylabel('Thrust Magnitude (N)')
    l1, b1 = ax.get_legend_handles_labels()
    l2, b2 = ax2b.get_legend_handles_labels()
    ax.legend(l1+l2, b1+b2, loc='upper right', fontsize=7)
    ax.set_title('Control Moments & Thrust')

    # ── Line-style legend ─────────────────────────────────────────────────────
    ax = axes[3, 0]
    ax.axis('off')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color='k', lw=2,  ls='-',  label='Actual (sensor)'),
        Line2D([0],[0], color='k', lw=1,  ls='--', label='Commanded (MPC x[1])'),
        Line2D([0],[0], color='k', lw=1.5,ls=':',  label='First-solve horizon'),
    ], loc='center', fontsize=11, frameon=True)
    ax.set_title('Line-style key')

    # ── Thrust magnitude detail ───────────────────────────────────────────────
    ax = axes[3, 1]
    ax.plot(cmd_t, cmd[:, C_TNORM], 'k-',  lw=2, label='Thrust (commanded)')
    if has_fs:
        ax.plot(fs_t, fs[:, F_TNORM], 'k:', lw=1.5, alpha=0.7, label='Thrust (init)')
    ax.set_ylabel('Thrust Magnitude (N)'); ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8); ax.grid(True); ax.set_title('Thrust Magnitude Detail')

    fig.suptitle(
        f'Flight Data: {os.path.basename(folder_path)}'
        + (f' (Cutoff: {cutoff_time}s)' if cutoff_time else ''),
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    suffix = f"_cutoff_{cutoff_time}s" if cutoff_time else ""
    # Save as PDF vector graphic (remove dpi)
    out = os.path.join(folder_path, f"comprehensive_plot{suffix}.pdf")
    plt.savefig(out, bbox_inches='tight')
    print(f"Comprehensive plot saved to: {out}")
    plt.show()


# ── 3D trajectory ─────────────────────────────────────────────────────────────
def plot_3d_trajectory(folder_path, cutoff_time=None):
    cmd = load_csv(os.path.join(folder_path, "commanded_state.csv"), cutoff_time)
    act = load_csv(os.path.join(folder_path, "actual_state.csv"),    cutoff_time)
    fs  = load_first_traj(os.path.join(folder_path, "first_solve_trajectory.csv"))

    if act is None:
        print("Error: could not load actual_state.csv"); return

    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection='3d')

    ax.plot(act[:, C_X], act[:, C_Y], act[:, C_Z], 'b-',  lw=2, label='Actual')
    if cmd is not None:
        ax.plot(cmd[:, C_X], cmd[:, C_Y], cmd[:, C_Z], 'r--', lw=1, alpha=0.7, label='Commanded')
    if fs is not None:
        ax.plot(fs[:, F_X], fs[:, F_Y], fs[:, F_Z], 'g:', lw=2, alpha=0.8, label='First-solve horizon')

    ax.scatter([act[0, C_X]],  [act[0, C_Y]],  [act[0, C_Z]],  c='green', s=100, marker='o', label='Start')
    ax.scatter([act[-1, C_X]], [act[-1, C_Y]], [act[-1, C_Z]], c='red',   s=100, marker='s', label='End')

    # Orientation quivers on actual trajectory
    n_arr = min(20, len(act))
    step  = max(1, len(act) // n_arr)
    alen  = 0.08
    for i in range(0, len(act), step):
        x, y, z = act[i, C_X], act[i, C_Y], act[i, C_Z]
        q = [act[i, C_QW], act[i, C_QX], act[i, C_QY], act[i, C_QZ]]
        for bv, col in zip([[alen,0,0],[0,alen,0],[0,0,alen]], ['r','g','b']):
            wv = quat_rotate(q, bv)
            ax.quiver(x, y, z, wv[0], wv[1], wv[2],
                      color=col, alpha=0.6, linewidth=1, arrow_length_ratio=0.3)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Trajectory: {os.path.basename(folder_path)}'
                 + (f' (Cutoff: {cutoff_time}s)' if cutoff_time else ''))

    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([],[],color='r',lw=2,label='Body X (Fwd)'),
          Line2D([],[],color='g',lw=2,label='Body Y (Left)'),
          Line2D([],[],color='b',lw=2,label='Body Z (Up)')]
    ax.legend(h, l + ['Body X (Fwd)','Body Y (Left)','Body Z (Up)'],
              loc='upper left', fontsize=8)
    ax.grid(True)

    ranges = np.array([act[:, C_X].ptp(), act[:, C_Y].ptp(), act[:, C_Z].ptp()])
    half   = max(ranges.max() / 2.0, 0.1)
    mids   = np.array([act[:, C_X].mean(), act[:, C_Y].mean(), act[:, C_Z].mean()])
    ax.set_xlim(mids[0]-half, mids[0]+half)
    ax.set_ylim(mids[1]-half, mids[1]+half)
    ax.set_zlim(mids[2]-half, mids[2]+half)

    plt.tight_layout()
    suffix = f"_cutoff_{cutoff_time}s" if cutoff_time else ""
    out = os.path.join(folder_path, f"3d_trajectory{suffix}.pdf")
    plt.savefig(out, bbox_inches='tight')
    print(f"3D trajectory plot saved to: {out}")
    plt.show()


# ── Error analysis ────────────────────────────────────────────────────────────
def plot_error_analysis(folder_path, cutoff_time=None):
    from scipy import interpolate

    cmd = load_csv(os.path.join(folder_path, "commanded_state.csv"), cutoff_time)
    act = load_csv(os.path.join(folder_path, "actual_state.csv"),    cutoff_time)

    if cmd is None or act is None:
        print("Error: could not load state files"); return

    cmd_t = cmd[:, C_TS] - cmd[0, C_TS]
    act_t = act[:, C_TS] - act[0, C_TS]

    # Interpolate commanded onto actual timestamps (state cols 1-13 only)
    cmd_interp = np.zeros((len(act_t), 14))
    cmd_interp[:, 0] = act[:, C_TS]
    for ci in range(1, 14):
        f = interpolate.interp1d(cmd_t, cmd[:, ci], kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
        cmd_interp[:, ci] = f(act_t)

    pos_errors     = act[:, 1:4]   - cmd_interp[:, 1:4]
    vel_errors     = act[:, 4:7]   - cmd_interp[:, 4:7]
    ang_vel_errors = act[:, 11:14] - cmd_interp[:, 11:14]

    def quat_angle_error(q1, q2):
        dot = np.abs(np.sum(q1 * q2, axis=1))
        return 2 * np.arccos(np.clip(dot, 0.0, 1.0))

    angle_err = quat_angle_error(cmd_interp[:, 7:11], act[:, 7:11])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for c, lbl, col in zip(['r','g','b'], ['X','Y','Z'], range(3)):
        ax.plot(act_t, pos_errors[:, col], c+'-', lw=2, label=f'{lbl} error')
    ax.axhline(0, color='k', ls=':', alpha=0.5)
    ax.set_ylabel('Position Error (m)'); ax.legend(); ax.grid(True)
    ax.set_title('Position Tracking Errors')

    ax = axes[0, 1]
    for c, lbl, col in zip(['r','g','b'], ['Vx','Vy','Vz'], range(3)):
        ax.plot(act_t, vel_errors[:, col], c+'-', lw=2, label=f'{lbl} error')
    ax.axhline(0, color='k', ls=':', alpha=0.5)
    ax.set_ylabel('Velocity Error (m/s)'); ax.legend(); ax.grid(True)
    ax.set_title('Velocity Tracking Errors')

    ax = axes[1, 0]
    ax.plot(act_t, np.degrees(angle_err), 'purple', lw=2)
    ax.axhline(0, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Orientation Error (deg)')
    ax.grid(True); ax.set_title('Orientation Error')

    ax = axes[1, 1]
    for c, lbl, col in zip(['r','g','b'], ['ωx','ωy','ωz'], range(3)):
        ax.plot(act_t, ang_vel_errors[:, col], c+'-', lw=2, label=f'{lbl} error')
    ax.axhline(0, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Angular Vel Error (rad/s)')
    ax.legend(); ax.grid(True); ax.set_title('Angular Velocity Errors')

    fig.suptitle(f'Tracking Errors: {os.path.basename(folder_path)}'
                 + (f' (Cutoff: {cutoff_time}s)' if cutoff_time else ''),
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    rms_pos = np.sqrt(np.mean(pos_errors**2, axis=0))
    rms_vel = np.sqrt(np.mean(vel_errors**2, axis=0))
    rms_ang = np.sqrt(np.mean(angle_err**2))
    rms_w   = np.sqrt(np.mean(ang_vel_errors**2, axis=0))
    print("\n=== RMS Tracking Errors ===")
    print(f"Position  : X={rms_pos[0]:.4f}m  Y={rms_pos[1]:.4f}m  Z={rms_pos[2]:.4f}m")
    print(f"Velocity  : Vx={rms_vel[0]:.4f}  Vy={rms_vel[1]:.4f}  Vz={rms_vel[2]:.4f} m/s")
    print(f"Orientation: {np.degrees(rms_ang):.4f} deg")
    print(f"Ang vel   : ωx={rms_w[0]:.4f}  ωy={rms_w[1]:.4f}  ωz={rms_w[2]:.4f} rad/s")

    suffix = f"_cutoff_{cutoff_time}s" if cutoff_time else ""
    out = os.path.join(folder_path, f"error_analysis{suffix}.pdf")
    plt.savefig(out, bbox_inches='tight')
    print(f"Error analysis plot saved to: {out}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────
def find_latest_flight_folder(logs_dir):
    folders = [p for p in Path(logs_dir).iterdir()
               if p.is_dir() and "_flight_" in p.name]
    return str(max(folders, key=lambda p: p.stat().st_mtime)) if folders else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot CoManDO MPC flight data')
    parser.add_argument('path', nargs='?', help='Flight folder or logs directory')
    parser.add_argument('--cutoff',          type=float, help='Cutoff time (seconds)')
    parser.add_argument('--latest',          action='store_true')
    parser.add_argument('--trajectory-only', action='store_true')
    parser.add_argument('--errors-only',     action='store_true')
    parser.add_argument('--all',             action='store_true')
    args = parser.parse_args()

    if args.latest:
        logs_dir = args.path or "./logs"
        if not os.path.exists(logs_dir):
            logs_dir = "../logs"
        folder_path = find_latest_flight_folder(logs_dir)
        if folder_path is None:
            print(f"Error: no flight folders found in {logs_dir}"); sys.exit(1)
        print(f"Using latest flight folder: {folder_path}")
    elif args.path:
        folder_path = args.path
        if not os.path.isdir(folder_path):
            print(f"Error: {folder_path} is not a directory"); sys.exit(1)
    else:
        print("Usage: python3 logger.py [path] [options]")
        print("  --cutoff TIME  --latest  --all  --trajectory-only  --errors-only")
        sys.exit(1)

    for req in ["commanded_state.csv", "actual_state.csv"]:
        if not os.path.exists(os.path.join(folder_path, req)):
            print(f"Error: {req} missing from {folder_path}"); sys.exit(1)

    if not os.path.exists(os.path.join(folder_path, "first_solve_trajectory.csv")):
        print("Note: first_solve_trajectory.csv not found — first-solve overlay disabled")

    print(f"Loading data from: {folder_path}")

    if args.trajectory_only:
        plot_3d_trajectory(folder_path, args.cutoff)
    elif args.errors_only:
        plot_error_analysis(folder_path, args.cutoff)
    elif args.all:
        plot_flight_data(folder_path, args.cutoff)
        plot_3d_trajectory(folder_path, args.cutoff)
        plot_error_analysis(folder_path, args.cutoff)
    else:
        plot_flight_data(folder_path, args.cutoff)

    print("\nDone!")