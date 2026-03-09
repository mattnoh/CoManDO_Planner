#!/usr/bin/env python3
"""
flight_analysis.py  —  MPC receding-horizon animation + flight report
──────────────────────────────────────────────────────────────────────
Outputs
  mpc_animation.mp4   –  2×2 grid, 4 orbital camera angles rotating
                         around the vertical axis; drone flies smoothly
  mpc_4view_static.pdf – snapshot of the 4-view scene
  flight_report.pdf   –  3D trajectory + grouped state comparison rows
                         (position / velocity / quaternion / ang-rates)

CSV layout (planner_node.cpp)
  all_solves.csv        solve_num, solve_time_ms, node, t,
                        x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz,
                        fz, mx,my,mz
  commanded_state.csv   timestamp, x,y,z, vx,vy,vz,
                        qw,qx,qy,qz, wx,wy,wz, fz,mx,my,mz
  actual_state.csv      timestamp, x,y,z, vx,vy,vz,
                        qw,qx,qy,qz, wx,wy,wz

Usage
  python3 flight_analysis.py <folder>
  python3 flight_analysis.py <folder> --fps 15 --fpsolve 8 --ghosts 10
  python3 flight_analysis.py <folder> --pdf-only | --anim-only
  python3 flight_analysis.py --latest [logs_dir]
  python3 flight_analysis.py <folder> --goal 0 0 1 --cutoff 10
──────────────────────────────────────────────────────────────────────
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from pathlib import Path

# ── Column indices ─────────────────────────────────────────────────────────────
AS_SOLVE=0; AS_TSOLVE=1; AS_NODE=2; AS_T=3
AS_X=4;  AS_Y=5;  AS_Z=6
AS_VX=7; AS_VY=8; AS_VZ=9
AS_QW=10; AS_QX=11; AS_QY=12; AS_QZ=13
AS_WX=14; AS_WY=15; AS_WZ=16

C_TS=0; C_X=1;  C_Y=2;  C_Z=3
C_VX=4; C_VY=5; C_VZ=6
C_QW=7; C_QX=8; C_QY=9; C_QZ=10
C_WX=11; C_WY=12; C_WZ=13

A_TS=0; A_X=1;  A_Y=2;  A_Z=3
A_VX=4; A_VY=5; A_VZ=6
A_QW=7; A_QX=8; A_QY=9; A_QZ=10
A_WX=11; A_WY=12; A_WZ=13

# ── Colours ────────────────────────────────────────────────────────────────────
CX = "#1D4ED8"; CY = "#16A34A"; CZ = "#DC2626"
CQW = "#7C3AED"; CQX = "#1D4ED8"; CQY = "#16A34A"; CQZ = "#DC2626"

CMD_ALPHA = 1.00
ACT_ALPHA = 0.70
FST_ALPHA = 0.42

FIRST_C = "#D97706"; GHOST_C = "#818CF8"; CURR_C = "#0891B2"
BF_X = "#EF4444";   BF_Y   = "#22C55E"; BF_Z   = "#3B82F6"
START_C = "#16A34A"; GOAL_C = "#9333EA"
ERR_C   = "#7C3AED"
GRID_C  = "#E5E7EB"; TEXT_C = "#111827"; DIM_C  = "#6B7280"
FILL_A  = 0.11

# ── 4 orbital views — same elevation, 90 deg apart ────────────────────────────
VIEWS = [
    dict(elev=22, azim_base=30,  label="View A"),
    dict(elev=22, azim_base=120, label="View B"),
    dict(elev=22, azim_base=210, label="View C"),
    dict(elev=12, azim_base=300, label="View D  (low)"),
]
AZIM_SPEED = 0.35   # deg/frame

# ── Global rcParams ────────────────────────────────────────────────────────────
FS = 12
plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#CBD5E1",
    "axes.grid":           True,
    "grid.color":          GRID_C,
    "grid.linewidth":      0.55,
    "grid.linestyle":      "--",
    "xtick.color":         DIM_C, "ytick.color": DIM_C,
    "xtick.labelsize":     FS-1,  "ytick.labelsize": FS-1,
    "font.size":           FS,
    "axes.labelsize":      FS,
    "axes.titlesize":      FS+1,
    "axes.labelcolor":     TEXT_C,
    "axes.titlecolor":     TEXT_C,
    "axes.titleweight":    "semibold",
    "legend.frameon":      True,
    "legend.edgecolor":    "#CBD5E1",
    "legend.facecolor":    "white",
    "legend.fontsize":     FS-1,
    "lines.linewidth":     2.0,
    "savefig.facecolor":   "white",
    "savefig.transparent": True,
    "pdf.fonttype":        42,   # embed TrueType → crisp at any zoom
    "ps.fonttype":         42,
})


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load(path, cutoff=None):
    try:
        d = np.genfromtxt(path, delimiter=",", skip_header=1)
        if d is None or d.ndim < 2 or len(d) == 0:
            return None
        if cutoff is not None:
            mask = d[:, 0] - d[0, 0] <= cutoff
            d    = d[mask] if mask.any() else d
        return d
    except Exception as e:
        print(f"  [warn] {os.path.basename(path)}: {e}")
        return None


def _find_latest(logs_dir):
    folders = [p for p in Path(logs_dir).iterdir()
               if p.is_dir() and any(k in p.name for k in
                  ("_flight_","_mpc_","_open_loop_","_landing_"))]
    return str(max(folders, key=lambda p: p.stat().st_mtime)) if folders else None


def _rel(d):  return d[:, 0] - d[0, 0]
def _trim(a, b):
    n = min(len(a), len(b)); return a[:n], b[:n]


def qmat_batch(qw, qx, qy, qz):
    q = np.stack([qw,qx,qy,qz],-1).astype(float)
    q /= np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
    N = len(w); R = np.zeros((N,3,3))
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-w*z);   R[:,0,2]=2*(x*z+w*y)
    R[:,1,0]=2*(x*y+w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-w*x)
    R[:,2,0]=2*(x*z-w*y);   R[:,2,1]=2*(y*z+w*x);   R[:,2,2]=1-2*(x*x+y*y)
    return R

def qmat1(qw,qx,qy,qz):
    return qmat_batch(np.array([qw]),np.array([qx]),
                      np.array([qy]),np.array([qz]))[0]

def att_err_deg(qw1,qx1,qy1,qz1, qw2,qx2,qy2,qz2):
    dot = np.abs(qw1*qw2+qx1*qx2+qy1*qy2+qz1*qz2)
    return np.degrees(2*np.arccos(np.clip(dot,0,1)))


def _style3d(ax, title=""):
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane):
        pane.fill=False; pane.set_edgecolor("#DDDDDD")
    ax.grid(True,color="#ECECEC",lw=0.4,ls="--")
    ax.set_rasterization_zorder(-1)          # lines stay vector in PDF
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.label.set_color(TEXT_C); axis.label.set_fontsize(FS-1)
        axis.set_tick_params(colors=DIM_C, labelsize=FS-2)
    ax.set_xlabel("x (m)",labelpad=3)
    ax.set_ylabel("y (m)",labelpad=3)
    ax.set_zlabel("z (m)",labelpad=3)
    if title: ax.set_title(title,fontsize=FS,color=TEXT_C,pad=5)


# ═══════════════════════════════════════════════════════════════════════════════
#  FlightData
# ═══════════════════════════════════════════════════════════════════════════════

class FlightData:
    def __init__(self, folder, cutoff=None):
        self.folder = folder

        raw = _load(os.path.join(folder,"all_solves.csv"),cutoff)
        if raw is None:
            raise FileNotFoundError("all_solves.csv not found / empty")

        self.cmd = _load(os.path.join(folder,"commanded_state.csv"),cutoff)
        self.act = _load(os.path.join(folder,"actual_state.csv"),   cutoff)

        solve_ids       = raw[:,AS_SOLVE].astype(int)
        self.solve_nums = np.unique(solve_ids)
        self.n_solves   = len(self.solve_nums)
        self.solve_map  = {s: raw[solve_ids==s] for s in self.solve_nums}

        self.drone_pos   = np.zeros((self.n_solves,3))
        self.drone_quat  = np.zeros((self.n_solves,4))
        self.solve_times = np.zeros(self.n_solves)
        for i,s in enumerate(self.solve_nums):
            traj = self.solve_map[s]
            r0   = traj[np.argmin(np.abs(traj[:,AS_NODE]))]
            self.drone_pos[i]   = r0[[AS_X,AS_Y,AS_Z]]
            self.drone_quat[i]  = r0[[AS_QW,AS_QX,AS_QY,AS_QZ]]
            self.solve_times[i] = r0[AS_TSOLVE]

        first        = self.solve_map[self.solve_nums[0]]
        self.start   = self.drone_pos[0].copy()
        last_row     = first[np.argmax(first[:,AS_NODE])]
        self.goal    = last_row[[AS_X,AS_Y,AS_Z]]
        self.n_nodes = int(first[:,AS_NODE].max())+1

        f0           = first[np.argsort(first[:,AS_NODE])]
        self.fs_t    = f0[:,AS_T]
        self.fs_xyz  = f0[:,[AS_X,AS_Y,AS_Z]]
        self.fs_vel  = f0[:,[AS_VX,AS_VY,AS_VZ]]
        self.fs_quat = f0[:,[AS_QW,AS_QX,AS_QY,AS_QZ]]
        self.fs_omg  = f0[:,[AS_WX,AS_WY,AS_WZ]]

        pad = 0.28
        xyz = raw[:,[AS_X,AS_Y,AS_Z]]
        self.xlim=(xyz[:,0].min()-pad, xyz[:,0].max()+pad)
        self.ylim=(xyz[:,1].min()-pad, xyz[:,1].max()+pad)
        self.zlim=(max(0,xyz[:,2].min()-pad), xyz[:,2].max()+pad)

    def horizon(self,si):
        s=self.solve_nums[si]; t=self.solve_map[s]
        t=t[np.argsort(t[:,AS_NODE])]
        return t[:,AS_X],t[:,AS_Y],t[:,AS_Z]

    def cmd_path(self,up_to):
        if self.cmd is not None:
            n=min(up_to+1,len(self.cmd)); return self.cmd[:n,[C_X,C_Y,C_Z]]
        return self.drone_pos[:up_to+1]

    def act_path(self,up_to):
        if self.act is not None:
            n=min(up_to+1,len(self.act)); return self.act[:n,[A_X,A_Y,A_Z]]
        return self.drone_pos[:up_to+1]


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF: flight_report.pdf
# ═══════════════════════════════════════════════════════════════════════════════

def save_pdf(data: FlightData, folder: str, goal_override=None):
    """
    Layout  (5 rows × 4 cols, GridSpec)
      Row 0  col 0:3 → 3D trajectory (wide)   col 3 → solve time
      Row 1  col 0:3 → Position xyz            col 3 → ‖Δp‖
      Row 2  col 0:3 → Velocity xyz            col 3 → ‖Δv‖
      Row 3  col 0:3 → Quaternion              col 3 → att error
      Row 4  col 0:3 → Angular rates           col 3 → summary
    """
    goal = np.array(goal_override) if goal_override else data.goal
    cmd  = data.cmd;  act = data.act
    has  = cmd is not None and act is not None
    if has:
        cmd_a,act_a = _trim(cmd,act); tc = _rel(cmd_a)

    fig = plt.figure(figsize=(22,28),facecolor="white")
    gs  = gridspec.GridSpec(5,4,hspace=0.60,wspace=0.36,
                             top=0.964,bottom=0.04,left=0.055,right=0.975)
    fig.suptitle(f"MPC Flight Report — {os.path.basename(folder)}",
                 fontsize=16,fontweight="bold",color=TEXT_C,y=0.984)

    # ── 3D ────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0,0:3],projection="3d")
    _style3d(ax3,"3-D Trajectory")
    ax3.set_xlim(data.xlim); ax3.set_ylim(data.ylim); ax3.set_zlim(data.zlim)

    n = data.n_solves
    for i in range(1,n):
        a  = 0.03+0.18*(i/max(n-1,1))
        lw = 0.4 +0.55*(i/max(n-1,1))
        ax3.plot(*data.horizon(i),color=GHOST_C,lw=lw,alpha=a)

    ax3.plot(*data.horizon(0),color=FIRST_C,lw=3.0,alpha=0.92,
             label="First-solve plan",zorder=8)

    if cmd is not None:
        ax3.plot(cmd[:,C_X],cmd[:,C_Y],cmd[:,C_Z],
                 color="#1D4ED8",lw=2.1,label="Commanded",zorder=10)
    if act is not None:
        ax3.plot(act[:,A_X],act[:,A_Y],act[:,A_Z],
                 color="#DC2626",lw=1.8,ls="--",alpha=0.80,
                 label="Actual",zorder=11)

    # Body frames at MPC timing (all solves, subsampled to ≤20)
    n_bf   = min(20,data.n_solves)
    bf_idx = np.linspace(0,data.n_solves-1,n_bf,dtype=int)
    extent = max(data.xlim[1]-data.xlim[0],
                 data.ylim[1]-data.ylim[0],
                 data.zlim[1]-data.zlim[0])
    BFS    = max(0.05,extent*0.055)
    Rs     = qmat_batch(data.drone_quat[bf_idx,0],data.drone_quat[bf_idx,1],
                        data.drone_quat[bf_idx,2],data.drone_quat[bf_idx,3])
    for k,fi in enumerate(bf_idx):
        p=data.drone_pos[fi]; R=Rs[k]
        for ci,col in enumerate([BF_X,BF_Y,BF_Z]):
            d=R[:,ci]*BFS
            ax3.plot([p[0],p[0]+d[0]],[p[1],p[1]+d[1]],[p[2],p[2]+d[2]],
                     color=col,lw=1.5,alpha=0.78,zorder=12)

    ax3.scatter(*data.start,s=110,color=START_C,marker="^",
                depthshade=False,zorder=15,label="Start")
    ax3.scatter(*goal,s=130,color=GOAL_C,marker="*",
                depthshade=False,zorder=15,label="Goal")
    ax3.legend(fontsize=FS-1,loc="upper left",
               facecolor="white",edgecolor="#CBD5E1",framealpha=0.92)

    # ── Solve time ─────────────────────────────────────────────────────────────
    ax_st = fig.add_subplot(gs[0,3])
    si = np.arange(data.n_solves)
    ax_st.bar(si,data.solve_times,color=GHOST_C,alpha=0.55,width=0.85)
    ax_st.plot(si,data.solve_times,color=GHOST_C,lw=1.0,alpha=0.7)
    mu  = data.solve_times.mean()
    p95 = np.percentile(data.solve_times,95)
    ax_st.axhline(mu, color=CURR_C,   ls="--",lw=1.6,label=f"μ = {mu:.1f} ms")
    ax_st.axhline(p95,color="#DC2626",ls=":", lw=1.4,label=f"p95 = {p95:.1f} ms")
    ax_st.set_title("Solve Time  (ms)")
    ax_st.set_xlabel("Solve index",fontsize=FS-1)
    ax_st.set_ylabel("ms",fontsize=FS-1)
    ax_st.legend(fontsize=FS-2)

    # ── Grouped state helper ───────────────────────────────────────────────────
    def _group(ax, cols_c, cols_a, labels, colors, unit, title,
               fs_cols=None, hline=None):
        for i,(cc,ca,lbl,col) in enumerate(zip(cols_c,cols_a,labels,colors)):
            ax.plot(tc,cmd_a[:,cc],color=col,lw=2.0,alpha=CMD_ALPHA,
                    label=f"{lbl}  cmd")
            ax.plot(tc,act_a[:,ca],color=col,lw=1.6,ls="--",alpha=ACT_ALPHA,
                    label=f"{lbl}  act")
            if fs_cols is not None:
                ax.plot(data.fs_t,fs_cols[:,i],color=col,lw=1.3,
                        ls="-.",alpha=FST_ALPHA,
                        label=f"{lbl}  plan₀")
        if hline is not None:
            ax.axhline(hline,color=DIM_C,ls=":",lw=1.0)
        ax.set_title(f"{title}  ({unit})" if unit else title)
        ax.set_xlabel("t  (s)",fontsize=FS-1)
        n_comp = len(labels)
        ax.legend(ncol=n_comp,fontsize=FS-2,loc="best",framealpha=0.88)

    def _enorm(ax, tc, err, title, unit, thr=None):
        ax.plot(tc,err,color=ERR_C,lw=2.0)
        ax.fill_between(tc,err,color=ERR_C,alpha=FILL_A)
        if thr is not None:
            ax.axhline(thr,color=DIM_C,ls=":",lw=1.2,label=f"thr {thr}")
            ax.legend(fontsize=FS-2)
        ax.set_title(f"{title}  ({unit})" if unit else title)
        ax.set_xlabel("t  (s)",fontsize=FS-1)

    if has:
        # Row 1: position
        _group(fig.add_subplot(gs[1,0:3]),
               [C_X,C_Y,C_Z],[A_X,A_Y,A_Z],
               ["x","y","z"],[CX,CY,CZ],"m",
               "Position  x / y / z",fs_cols=data.fs_xyz)
        pe = np.linalg.norm(cmd_a[:,[C_X,C_Y,C_Z]]-act_a[:,[A_X,A_Y,A_Z]],axis=1)
        _enorm(fig.add_subplot(gs[1,3]),tc,pe,"‖Δp‖","m",thr=0.05)

        # Row 2: velocity
        _group(fig.add_subplot(gs[2,0:3]),
               [C_VX,C_VY,C_VZ],[A_VX,A_VY,A_VZ],
               ["vx","vy","vz"],[CX,CY,CZ],"m/s",
               "Velocity  vx / vy / vz",fs_cols=data.fs_vel,hline=0.0)
        ve = np.linalg.norm(cmd_a[:,[C_VX,C_VY,C_VZ]]-act_a[:,[A_VX,A_VY,A_VZ]],axis=1)
        _enorm(fig.add_subplot(gs[2,3]),tc,ve,"‖Δv‖","m/s")

        # Row 3: quaternion
        _group(fig.add_subplot(gs[3,0:3]),
               [C_QW,C_QX,C_QY,C_QZ],[A_QW,A_QX,A_QY,A_QZ],
               ["qw","qx","qy","qz"],[CQW,CQX,CQY,CQZ],"",
               "Quaternion  qw / qx / qy / qz",fs_cols=data.fs_quat)
        ae = att_err_deg(cmd_a[:,C_QW],cmd_a[:,C_QX],
                          cmd_a[:,C_QY],cmd_a[:,C_QZ],
                          act_a[:,A_QW],act_a[:,A_QX],
                          act_a[:,A_QY],act_a[:,A_QZ])
        _enorm(fig.add_subplot(gs[3,3]),tc,ae,"Att error","°",thr=5.0)

        # Row 4: angular rates
        _group(fig.add_subplot(gs[4,0:3]),
               [C_WX,C_WY,C_WZ],[A_WX,A_WY,A_WZ],
               ["ωx","ωy","ωz"],[CX,CY,CZ],"rad/s",
               "Angular Rate  ωx / ωy / ωz",fs_cols=data.fs_omg,hline=0.0)

        # Summary
        ax_s = fig.add_subplot(gs[4,3]); ax_s.axis("off")
        txt = (
            f"Position\n"
            f"  max   {pe.max():.4f} m\n"
            f"  mean  {pe.mean():.4f} m\n"
            f"  final {pe[-1]:.4f} m\n\n"
            f"Velocity\n"
            f"  max   {ve.max():.4f} m/s\n"
            f"  mean  {ve.mean():.4f} m/s\n\n"
            f"Attitude\n"
            f"  max   {ae.max():.3f}°\n"
            f"  mean  {ae.mean():.3f}°\n\n"
            f"Solver\n"
            f"  mean  {mu:.2f} ms\n"
            f"  p95   {p95:.2f} ms\n"
            f"  max   {data.solve_times.max():.2f} ms\n\n"
            f"Solves:  {data.n_solves}\n"
            f"Horizon: {data.n_nodes} nodes"
        )
        ax_s.text(0.06,0.96,txt,transform=ax_s.transAxes,
                  va="top",fontsize=FS,fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.6",
                            facecolor="#F0F7FF",edgecolor="#BFDBFE",alpha=0.95))
    else:
        fig.text(0.5,0.50,"commanded_state.csv or actual_state.csv missing",
                 ha="center",va="center",fontsize=14,color=DIM_C)

    out = os.path.join(folder,"flight_report.pdf")
    fig.savefig(out,bbox_inches="tight",dpi=300,
                transparent=True,format="pdf")
    print(f"  Saved: {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Animation
# ═══════════════════════════════════════════════════════════════════════════════

class MPCAnimator:
    """
    2×2 grid — 4 orbital cameras rotating around vertical axis.
    Drone interpolates continuously between solve positions.
    No phase stops — smooth from start to end.
    """
    def __init__(self, data: FlightData, fps=12, fpsolve=8,
                 n_ghosts=8, goal_override=None):
        self.data    = data
        self.fps     = fps
        self.fpsolve = fpsolve
        self.n_ghosts= n_ghosts
        self.goal    = np.array(goal_override) if goal_override else data.goal
        self.total_frames = data.n_solves * fpsolve

        extent = max(data.xlim[1]-data.xlim[0],
                     data.ylim[1]-data.ylim[0],
                     data.zlim[1]-data.zlim[0])
        self.BFS = max(0.06,extent*0.11)

        # ── Figure ──────────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(18,12),facecolor="white")
        self.fig.patch.set_facecolor("white")
        self.fig.subplots_adjust(left=0.03,right=0.84,top=0.92,bottom=0.03,
                                  hspace=0.08,wspace=0.06)

        self.axes = []
        for vi,view in enumerate(VIEWS):
            ax = self.fig.add_subplot(2,2,vi+1,projection="3d")
            _style3d(ax,view["label"])
            ax.set_xlim(data.xlim); ax.set_ylim(data.ylim); ax.set_zlim(data.zlim)
            ax.view_init(elev=view["elev"],azim=view["azim_base"])
            self.axes.append(ax)

        # ── Static artists ───────────────────────────────────────────────────────
        fx0,fy0,fz0 = data.horizon(0)
        self._first_lines = []
        for ax in self.axes:
            ln, = ax.plot(fx0,fy0,fz0,color=FIRST_C,lw=2.8,alpha=0.90,zorder=6,
                          label="First-solve plan")
            self._first_lines.append(ln)
            ax.scatter(*data.start,s=90,color=START_C,marker="^",
                       depthshade=False,zorder=15)
            ax.scatter(*self.goal, s=110,color=GOAL_C, marker="*",
                       depthshade=False,zorder=15)

        # ── Dynamic artists ──────────────────────────────────────────────────────
        self._ghosts   = [[None]*n_ghosts for _ in range(4)]
        self._curr     = [None]*4
        self._cmd_line = [None]*4
        self._act_line = [None]*4
        self._drone    = [None]*4
        self._bflines  = [[None,None,None] for _ in range(4)]

        for vi,ax in enumerate(self.axes):
            for k in range(n_ghosts):
                ln, = ax.plot([],[],[],color=GHOST_C,lw=0.6,alpha=0.0)
                self._ghosts[vi][k] = ln
            self._curr[vi],    = ax.plot([],[],[],color=CURR_C,lw=2.4,
                                          alpha=0.92,zorder=7,
                                          label="Current plan" if vi==0 else "_")
            self._cmd_line[vi],= ax.plot([],[],[],color="#1D4ED8",lw=1.9,
                                          zorder=10,
                                          label="Commanded" if vi==0 else "_")
            self._act_line[vi],= ax.plot([],[],[],color="#DC2626",lw=1.7,
                                          ls="--",alpha=0.80,zorder=11,
                                          label="Actual" if vi==0 else "_")
            self._drone[vi],   = ax.plot([],[],[],"o",color=CURR_C,ms=9,
                                          markeredgecolor="white",
                                          markeredgewidth=1.5,zorder=14)
            for ci,col in enumerate([BF_X,BF_Y,BF_Z]):
                ln, = ax.plot([],[],[],color=col,lw=2.0,alpha=0.90,zorder=13)
                self._bflines[vi][ci] = ln

        self.axes[0].legend(fontsize=FS-2,loc="upper left",
                            facecolor="white",edgecolor="#CBD5E1",framealpha=0.90)

        # ── Info + legend panel ───────────────────────────────────────────────────
        self._info = self.fig.text(
            0.862,0.78,"",color=TEXT_C,fontsize=FS-1,va="top",ha="left",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.55",facecolor="white",
                      edgecolor="#CBD5E1",alpha=0.94))

        self.fig.text(0.5,0.97,
                      f"Receding-Horizon MPC — {os.path.basename(data.folder)}",
                      ha="center",va="top",fontsize=FS+2,
                      fontweight="bold",color=TEXT_C)

        self.fig.text(0.863,0.42,"Body frame",color=DIM_C,
                      fontsize=FS-2,va="top")
        for ci,(col,lbl) in enumerate([(BF_X,"X"),(BF_Y,"Y"),(BF_Z,"Z")]):
            self.fig.text(0.863,0.39-ci*0.035,f"  ■ {lbl}",
                          color=col,fontsize=FS-2,va="top",fontweight="bold")

        self._prev_solve = -1

    # ── per-frame helpers ────────────────────────────────────────────────────────

    def _set_bf(self,vi,pos,R):
        for ci,ln in enumerate(self._bflines[vi]):
            d = R[:,ci]*self.BFS
            ln.set_data_3d([pos[0],pos[0]+d[0]],
                            [pos[1],pos[1]+d[1]],
                            [pos[2],pos[2]+d[2]])

    def _set_ghosts(self,vi,si):
        glist = list(range(max(0,si-self.n_ghosts),si))
        ng    = len(glist)
        for k,ln in enumerate(self._ghosts[vi]):
            if k < ng:
                gi    = glist[k]
                frac  = (k+1)/max(ng,1)
                ln.set_data_3d(*self.data.horizon(gi))
                ln.set_linewidth(0.5+1.2*frac)
                ln.set_alpha(0.04+0.26*frac)
            else:
                ln.set_alpha(0.0)

    # ── main update ───────────────────────────────────────────────────────────────

    def update(self,frame):
        data    = self.data
        solve_i = min(frame//self.fpsolve, data.n_solves-1)
        alpha   = (frame % self.fpsolve) / max(self.fpsolve-1,1)
        nxt     = min(solve_i+1, data.n_solves-1)

        pos  = (1-alpha)*data.drone_pos[solve_i] + alpha*data.drone_pos[nxt]
        quat = data.drone_quat[solve_i] if alpha<0.5 else data.drone_quat[nxt]
        R    = qmat1(*quat)

        cp = data.cmd_path(solve_i)
        ap = data.act_path(solve_i)

        if solve_i != self._prev_solve:
            hx,hy,hz = data.horizon(solve_i)
            for vi in range(4):
                self._curr[vi].set_data_3d(hx,hy,hz)
                self._set_ghosts(vi,solve_i)
            self._prev_solve = solve_i

        for vi in range(4):
            azim = VIEWS[vi]["azim_base"] + frame*AZIM_SPEED
            self.axes[vi].view_init(elev=VIEWS[vi]["elev"],azim=azim)
            self._drone[vi].set_data_3d([pos[0]],[pos[1]],[pos[2]])
            self._set_bf(vi,pos,R)
            self._cmd_line[vi].set_data_3d(cp[:,0],cp[:,1],cp[:,2])
            self._act_line[vi].set_data_3d(ap[:,0],ap[:,1],ap[:,2])

        dist = np.linalg.norm(pos-self.goal)
        self._info.set_text(
            f"Solve : {solve_i+1:>4d} / {data.n_solves}\n"
            f"t_sol : {data.solve_times[solve_i]:>6.1f} ms\n"
            f"pos   : [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]\n"
            f"‖Δgoal‖: {dist:.3f} m\n"
            f"azim  : {VIEWS[0]['azim_base']+frame*AZIM_SPEED:.1f}°"
        )

        artists = [self._info]
        for vi in range(4):
            artists += ([self._curr[vi],self._cmd_line[vi],
                         self._act_line[vi],self._drone[vi]]
                        + self._ghosts[vi]+self._bflines[vi])
        return artists

    # ── save ──────────────────────────────────────────────────────────────────────

    def save(self,out_path):
        print(f"  Rendering {self.total_frames} frames "
              f"({self.data.n_solves} solves × {self.fpsolve} frames/solve) …")
        anim = animation.FuncAnimation(
            self.fig,self.update,
            frames=self.total_frames,
            interval=1000//self.fps,
            blit=False,repeat=False)
        try:
            writer = animation.FFMpegWriter(
                fps=self.fps,bitrate=3000,
                extra_args=["-vcodec","libx264","-pix_fmt","yuv420p","-crf","18"])
            anim.save(out_path,writer=writer,dpi=120,
                      savefig_kwargs={"facecolor":"white"})
            print(f"  Saved (MP4): {out_path}")
        except Exception as e:
            gif = out_path.replace(".mp4",".gif")
            print(f"  [warn] ffmpeg ({e}) — GIF fallback: {gif}")
            anim.save(gif,writer=animation.PillowWriter(fps=self.fps),
                      dpi=90,savefig_kwargs={"facecolor":"white"})
            print(f"  Saved (GIF): {gif}")
        plt.close(self.fig)

    def save_static(self,out_path):
        self.update(0)
        self.fig.savefig(out_path,bbox_inches="tight",dpi=300,
                         transparent=True,format="pdf",facecolor="white")
        print(f"  Saved (4-view static): {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="MPC flight analysis")
    ap.add_argument("path",      nargs="?")
    ap.add_argument("--cutoff",  type=float)
    ap.add_argument("--fps",     type=int, default=12)
    ap.add_argument("--fpsolve", type=int, default=8,
                    help="Anim frames per solve step [default: 8]")
    ap.add_argument("--ghosts",  type=int, default=8)
    ap.add_argument("--latest",  action="store_true")
    ap.add_argument("--pdf-only",  action="store_true")
    ap.add_argument("--anim-only", action="store_true")
    ap.add_argument("--goal", type=float, nargs=3, metavar=("X","Y","Z"))
    args = ap.parse_args()

    if args.latest:
        logs_dir = args.path or "./logs"
        if not os.path.isdir(logs_dir): logs_dir = "../logs"
        folder = _find_latest(logs_dir)
        if not folder: print(f"ERROR: no flight folders in {logs_dir}"); sys.exit(1)
        print(f"Latest: {folder}")
    elif args.path and os.path.isdir(args.path):
        folder = args.path
    else:
        ap.print_help(); sys.exit(1)

    print(f"\nLoading: {folder}")
    try:
        data = FlightData(folder, args.cutoff)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); sys.exit(1)

    print(f"  {data.n_solves} solves | {data.n_nodes} nodes/horizon")
    print(f"  start {data.start}  goal {data.goal}")
    if data.cmd is not None: print(f"  commanded_state: {len(data.cmd)} rows")
    if data.act is not None: print(f"  actual_state:    {len(data.act)} rows")

    if not args.anim_only:
        print("\nGenerating flight_report.pdf …")
        save_pdf(data, folder, args.goal)

    if not args.pdf_only:
        print("\nBuilding animation …")
        anim = MPCAnimator(data, fps=args.fps, fpsolve=args.fpsolve,
                           n_ghosts=args.ghosts, goal_override=args.goal)
        anim.save_static(os.path.join(folder,"mpc_4view_static.pdf"))
        anim.save(os.path.join(folder,"mpc_animation.mp4"))

    print("\nDone.")

if __name__ == "__main__":
    main()