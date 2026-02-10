#!/usr/bin/env python3
"""
Plot logged flight data from CoManDO MPC planner

Usage:
    python3 plot_flight_logs.py /path/to/flight_folder [cutoff_time]

This will look for CSV files inside the flight folder:
    - commanded_state.csv
    - actual_state.csv
    - control.csv

Optional cutoff_time: Only plot data up to this time (in seconds)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_csv(filepath, cutoff_time=None):
    """Load CSV file, return data as numpy array with optional cutoff"""
    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data is None or data.shape[0] == 0:
            return None
        
        # Apply cutoff if specified
        if cutoff_time is not None:
            # Get timestamps (first column) and convert to relative time
            timestamps = data[:, 0] - data[0, 0]
            # Find indices where time <= cutoff_time
            indices = timestamps <= cutoff_time
            if np.any(indices):
                data = data[indices]
            else:
                print(f"Warning: cutoff_time {cutoff_time}s is before all data in {filepath}")
                return None
        
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def plot_flight_data(folder_path, cutoff_time=None):
    """Plot all flight data from the given flight folder"""
    
    # Construct file paths
    cmd_state_path = os.path.join(folder_path, "commanded_state.csv")
    actual_state_path = os.path.join(folder_path, "actual_state.csv")
    control_path = os.path.join(folder_path, "control.csv")
    
    # Load data files with optional cutoff
    cmd_state = load_csv(cmd_state_path, cutoff_time)
    actual_state = load_csv(actual_state_path, cutoff_time)
    control = load_csv(control_path, cutoff_time)
    
    if cmd_state is None or actual_state is None or control is None:
        print("Error: Could not load one or more data files")
        print(f"  Commanded state: {cmd_state_path}")
        print(f"  Actual state: {actual_state_path}")
        print(f"  Control: {control_path}")
        return
    
    # Extract timestamps (relative to start)
    cmd_t = cmd_state[:, 0] - cmd_state[0, 0]
    actual_t = actual_state[:, 0] - actual_state[0, 0]
    ctrl_t = control[:, 0] - control[0, 0]
    
    # Create figure with 6 subplots (3x2 layout)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # ========== POSITION ==========
    ax1 = axes[0, 0]
    ax1.plot(actual_t, actual_state[:, 1], 'r-', label='Actual X', linewidth=2)
    ax1.plot(actual_t, actual_state[:, 2], 'g-', label='Actual Y', linewidth=2)
    ax1.plot(actual_t, actual_state[:, 3], 'b-', label='Actual Z', linewidth=2)
    ax1.plot(cmd_t, cmd_state[:, 1], 'r--', label='Cmd X', linewidth=1, alpha=0.7)
    ax1.plot(cmd_t, cmd_state[:, 2], 'g--', label='Cmd Y', linewidth=1, alpha=0.7)
    ax1.plot(cmd_t, cmd_state[:, 3], 'b--', label='Cmd Z', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Position (m)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True)
    ax1.set_title('Position Tracking')
    
    # ========== VELOCITY ==========
    ax2 = axes[0, 1]
    ax2.plot(actual_t, actual_state[:, 4], 'r-', label='Actual Vx', linewidth=2)
    ax2.plot(actual_t, actual_state[:, 5], 'g-', label='Actual Vy', linewidth=2)
    ax2.plot(actual_t, actual_state[:, 6], 'b-', label='Actual Vz', linewidth=2)
    ax2.plot(cmd_t, cmd_state[:, 4], 'r--', label='Cmd Vx', linewidth=1, alpha=0.7)
    ax2.plot(cmd_t, cmd_state[:, 5], 'g--', label='Cmd Vy', linewidth=1, alpha=0.7)
    ax2.plot(cmd_t, cmd_state[:, 6], 'b--', label='Cmd Vz', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Velocity (m/s)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True)
    ax2.set_title('Velocity Tracking')
    
    # ========== QUATERNIONS ==========
    ax3 = axes[1, 0]
    ax3.plot(actual_t, actual_state[:, 7], 'r-', label='Actual qw', linewidth=2)
    ax3.plot(actual_t, actual_state[:, 8], 'g-', label='Actual qx', linewidth=2)
    ax3.plot(actual_t, actual_state[:, 9], 'b-', label='Actual qy', linewidth=2)
    ax3.plot(actual_t, actual_state[:, 10], 'orange', label='Actual qz', linewidth=2)
    ax3.plot(cmd_t, cmd_state[:, 7], 'r--', label='Cmd qw', linewidth=1, alpha=0.7)
    ax3.plot(cmd_t, cmd_state[:, 8], 'g--', label='Cmd qx', linewidth=1, alpha=0.7)
    ax3.plot(cmd_t, cmd_state[:, 9], 'b--', label='Cmd qy', linewidth=1, alpha=0.7)
    ax3.plot(cmd_t, cmd_state[:, 10], 'orange', linestyle='--', label='Cmd qz', linewidth=1, alpha=0.7)
    ax3.set_ylabel('Quaternion')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True)
    ax3.set_title('Orientation (Quaternions)')
    
    # ========== ANGULAR VELOCITY ==========
    ax4 = axes[1, 1]
    ax4.plot(actual_t, actual_state[:, 11], 'r-', label='Actual ωx', linewidth=2)
    ax4.plot(actual_t, actual_state[:, 12], 'g-', label='Actual ωy', linewidth=2)
    ax4.plot(actual_t, actual_state[:, 13], 'b-', label='Actual ωz', linewidth=2)
    ax4.plot(cmd_t, cmd_state[:, 11], 'r--', label='Cmd ωx', linewidth=1, alpha=0.7)
    ax4.plot(cmd_t, cmd_state[:, 12], 'g--', label='Cmd ωy', linewidth=1, alpha=0.7)
    ax4.plot(cmd_t, cmd_state[:, 13], 'b--', label='Cmd ωz', linewidth=1, alpha=0.7)
    ax4.set_ylabel('Angular Velocity (rad/s)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True)
    ax4.set_title('Angular Velocity')
    
    # ========== CONTROL FORCES ==========
    ax5 = axes[2, 0]
    ax5.plot(ctrl_t, control[:, 1], 'r-', label='Fx', linewidth=2)
    ax5.plot(ctrl_t, control[:, 2], 'g-', label='Fy', linewidth=2)
    ax5.plot(ctrl_t, control[:, 3], 'b-', label='Fz', linewidth=2)
    ax5.set_ylabel('Force (N)')
    ax5.set_xlabel('Time (s)')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True)
    ax5.set_title('Control Forces')
    
    # ========== CONTROL MOMENTS & THRUST ==========
    ax6 = axes[2, 1]
    # Plot moments on left y-axis
    ax6.plot(ctrl_t, control[:, 4], 'r-', label='Mx', linewidth=2)
    ax6.plot(ctrl_t, control[:, 5], 'g-', label='My', linewidth=2)
    ax6.plot(ctrl_t, control[:, 6], 'b-', label='Mz', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Moment (Nm)', color='black')
    ax6.tick_params(axis='y', labelcolor='black')
    ax6.legend(loc='upper left', fontsize=8)
    ax6.grid(True)
    
    # Create second y-axis for thrust magnitude
    ax6b = ax6.twinx()
    ax6b.plot(ctrl_t, control[:, 7], 'k-', label='Thrust Magnitude', linewidth=2, alpha=0.8)
    ax6b.set_ylabel('Thrust Magnitude (N)', color='black')
    ax6b.tick_params(axis='y', labelcolor='black')
    
    # Combine legends from both axes
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax6.set_title('Control Moments & Thrust')
    
    # Add overall title
    fig.suptitle(f'Flight Data: {os.path.basename(folder_path)}' + 
                (f' (Cutoff: {cutoff_time}s)' if cutoff_time else ''), 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    
    # Save figure inside the flight folder
    if cutoff_time:
        output_filename = f"comprehensive_plot_cutoff_{cutoff_time}s.png"
    else:
        output_filename = "comprehensive_plot.png"
    
    output_path = os.path.join(folder_path, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comprehensive plot saved to: {output_path}")
    
    # Show plot
    plt.show()


def plot_3d_trajectory(folder_path, cutoff_time=None):
    """Plot 3D trajectory with orientation visualization"""
    
    # Construct file paths
    cmd_state_path = os.path.join(folder_path, "commanded_state.csv")
    actual_state_path = os.path.join(folder_path, "actual_state.csv")
    
    # Load data with optional cutoff
    cmd_state = load_csv(cmd_state_path, cutoff_time)
    actual_state = load_csv(actual_state_path, cutoff_time)
    
    if cmd_state is None or actual_state is None:
        print("Error: Could not load state files")
        return
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(actual_state[:, 1], actual_state[:, 2], actual_state[:, 3], 
            'b-', label='Actual Trajectory', linewidth=2)
    ax.plot(cmd_state[:, 1], cmd_state[:, 2], cmd_state[:, 3], 
            'r--', label='Commanded Trajectory', linewidth=1, alpha=0.7)
    
    # Plot start and end points
    ax.scatter([actual_state[0, 1]], [actual_state[0, 2]], [actual_state[0, 3]], 
               c='green', s=100, marker='o', label='Start')
    ax.scatter([actual_state[-1, 1]], [actual_state[-1, 2]], [actual_state[-1, 3]], 
               c='red', s=100, marker='s', label='End')
    
    # Add orientation arrows at key points
    num_orientation_arrows = min(20, len(actual_state))
    step = max(1, len(actual_state) // num_orientation_arrows)
    
    for i in range(0, len(actual_state), step):
        if i >= len(actual_state):
            break
            
        # Get position and quaternion
        x, y, z = actual_state[i, 1:4]
        qw, qx, qy, qz = actual_state[i, 7:11]
        
        # Convert quaternion to rotation matrix
        # We'll use a simple approach: body axes in world frame
        # Scale of orientation arrows
        arrow_length = 0.1
        
        # For a simple visualization, we'll use the quaternion to rotate the body axes
        # Body axes in body frame: X forward, Y left, Z up
        body_x = np.array([arrow_length, 0, 0])
        body_y = np.array([0, arrow_length, 0])
        body_z = np.array([0, 0, arrow_length])
        
        # Rotate body axes to world frame using quaternion
        def quaternion_rotate(q, v):
            qw, qx, qy, qz = q
            vx, vy, vz = v
            # Calculate quaternion product: q * v * q_conjugate
            t2 =  qw*qx
            t3 =  qw*qy
            t4 =  qw*qz
            t5 = -qx*qx
            t6 =  qx*qy
            t7 =  qx*qz
            t8 = -qy*qy
            t9 =  qy*qz
            t10= -qz*qz
            return np.array([
                2*( (t8 + t10)*vx + (t6 - t4)*vy + (t3 + t7)*vz ) + vx,
                2*( (t4 + t6)*vx + (t5 + t10)*vy + (t9 - t2)*vz ) + vy,
                2*( (t7 - t3)*vx + (t2 + t9)*vy + (t5 + t8)*vz ) + vz
            ])
        
        # Rotate body axes
        world_x = quaternion_rotate([qw, qx, qy, qz], body_x)
        world_y = quaternion_rotate([qw, qx, qy, qz], body_y)
        world_z = quaternion_rotate([qw, qx, qy, qz], body_z)
        
        # Plot orientation arrows
        ax.quiver(x, y, z, world_x[0], world_x[1], world_x[2], 
                 color='r', alpha=0.7, linewidth=1, arrow_length_ratio=0.3)
        ax.quiver(x, y, z, world_y[0], world_y[1], world_y[2], 
                 color='g', alpha=0.7, linewidth=1, arrow_length_ratio=0.3)
        ax.quiver(x, y, z, world_z[0], world_z[1], world_z[2], 
                 color='b', alpha=0.7, linewidth=1, arrow_length_ratio=0.3)
    
    # Create a custom legend for the orientation arrows
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs
        
        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)
    
    # Add orientation legend
    orientation_legend = [
        ax.plot([], [], [], 'r-', linewidth=2, label='Body X (Forward)')[0],
        ax.plot([], [], [], 'g-', linewidth=2, label='Body Y (Left)')[0],
        ax.plot([], [], [], 'b-', linewidth=2, label='Body Z (Up)')[0]
    ]
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Trajectory with Orientation: {os.path.basename(folder_path)}' + 
                (f' (Cutoff: {cutoff_time}s)' if cutoff_time else ''))
    
    # Combine all legends
    handles, labels = ax.get_legend_handles_labels()
    handles = handles + orientation_legend
    labels = labels + ['Body X (Forward)', 'Body Y (Left)', 'Body Z (Up)']
    ax.legend(handles, labels, loc='upper left')
    
    ax.grid(True)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([
        actual_state[:, 1].max()-actual_state[:, 1].min(),
        actual_state[:, 2].max()-actual_state[:, 2].min(),
        actual_state[:, 3].max()-actual_state[:, 3].min()
    ]).max() / 2.0
    
    mid_x = (actual_state[:, 1].max()+actual_state[:, 1].min()) * 0.5
    mid_y = (actual_state[:, 2].max()+actual_state[:, 2].min()) * 0.5
    mid_z = (actual_state[:, 3].max()+actual_state[:, 3].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    # Save figure inside the flight folder
    if cutoff_time:
        output_filename = f"3d_trajectory_cutoff_{cutoff_time}s.png"
    else:
        output_filename = "3d_trajectory.png"
    
    output_path = os.path.join(folder_path, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"3D trajectory plot saved to: {output_path}")
    
    plt.show()
def plot_error_analysis(folder_path, cutoff_time=None):
    """Plot position and orientation errors"""
    
    # Construct file paths
    cmd_state_path = os.path.join(folder_path, "commanded_state.csv")
    actual_state_path = os.path.join(folder_path, "actual_state.csv")
    
    # Load data with optional cutoff
    cmd_state = load_csv(cmd_state_path, cutoff_time)
    actual_state = load_csv(actual_state_path, cutoff_time)
    
    if cmd_state is None or actual_state is None:
        print("Error: Could not load state files")
        return
    
    # Interpolate commanded data to match actual timestamps
    from scipy import interpolate
    
    cmd_t = cmd_state[:, 0] - cmd_state[0, 0]
    actual_t = actual_state[:, 0] - actual_state[0, 0]
    
    # Create interpolation functions for commanded states
    interp_functions = []
    for i in range(1, cmd_state.shape[1]):  # Skip timestamp column
        f = interpolate.interp1d(cmd_t, cmd_state[:, i], kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
        interp_functions.append(f)
    
    # Interpolate commanded states at actual timestamps
    cmd_interp = np.zeros_like(actual_state)
    cmd_interp[:, 0] = actual_state[:, 0]  # Keep original timestamps
    for i in range(1, cmd_state.shape[1]):
        cmd_interp[:, i] = interp_functions[i-1](actual_t)
    
    # Calculate errors
    pos_errors = actual_state[:, 1:4] - cmd_interp[:, 1:4]  # x,y,z errors
    vel_errors = actual_state[:, 4:7] - cmd_interp[:, 4:7]  # vx,vy,vz errors
    
    # Calculate quaternion error (angle difference)
    # For quaternions, we can calculate the angular error
    def quaternion_angle_error(q1, q2):
        # Calculate the rotation from q1 to q2: q_err = q2 * q1^-1
        # For small errors, the angle is 2 * acos(|q_err.w|)
        q_err_w = np.abs(q1[:, 0]*q2[:, 0] + q1[:, 1]*q2[:, 1] + 
                        q1[:, 2]*q2[:, 2] + q1[:, 3]*q2[:, 3])
        q_err_w = np.clip(q_err_w, -1.0, 1.0)
        return 2 * np.arccos(q_err_w)  # in radians
    
    actual_q = actual_state[:, 7:11]  # qw, qx, qy, qz
    cmd_q = cmd_interp[:, 7:11]
    angle_errors = quaternion_angle_error(cmd_q, actual_q)
    
    # Angular velocity errors
    ang_vel_errors = actual_state[:, 11:14] - cmd_interp[:, 11:14]  # wx,wy,wz errors
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position errors
    ax1 = axes[0, 0]
    ax1.plot(actual_t, pos_errors[:, 0], 'r-', label='X error', linewidth=2)
    ax1.plot(actual_t, pos_errors[:, 1], 'g-', label='Y error', linewidth=2)
    ax1.plot(actual_t, pos_errors[:, 2], 'b-', label='Z error', linewidth=2)
    ax1.axhline(y=0.0, color='k', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Position Error (m)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Position Tracking Errors')
    
    # Velocity errors
    ax2 = axes[0, 1]
    ax2.plot(actual_t, vel_errors[:, 0], 'r-', label='Vx error', linewidth=2)
    ax2.plot(actual_t, vel_errors[:, 1], 'g-', label='Vy error', linewidth=2)
    ax2.plot(actual_t, vel_errors[:, 2], 'b-', label='Vz error', linewidth=2)
    ax2.axhline(y=0.0, color='k', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Velocity Error (m/s)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Velocity Tracking Errors')
    
    # Orientation error (angle)
    ax3 = axes[1, 0]
    ax3.plot(actual_t, np.degrees(angle_errors), 'purple', linewidth=2)
    ax3.axhline(y=0.0, color='k', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Orientation Error (deg)')
    ax3.grid(True)
    ax3.set_title('Orientation Error (Angle)')
    
    # Angular velocity errors
    ax4 = axes[1, 1]
    ax4.plot(actual_t, ang_vel_errors[:, 0], 'r-', label='ωx error', linewidth=2)
    ax4.plot(actual_t, ang_vel_errors[:, 1], 'g-', label='ωy error', linewidth=2)
    ax4.plot(actual_t, ang_vel_errors[:, 2], 'b-', label='ωz error', linewidth=2)
    ax4.axhline(y=0.0, color='k', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular Vel Error (rad/s)')
    ax4.legend()
    ax4.grid(True)
    ax4.set_title('Angular Velocity Errors')
    
    plt.suptitle(f'Tracking Error Analysis: {os.path.basename(folder_path)}' + 
                (f' (Cutoff: {cutoff_time}s)' if cutoff_time else ''), 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Calculate and print RMS errors
    rms_pos = np.sqrt(np.mean(pos_errors**2, axis=0))
    rms_vel = np.sqrt(np.mean(vel_errors**2, axis=0))
    rms_angle = np.sqrt(np.mean(angle_errors**2))
    rms_ang_vel = np.sqrt(np.mean(ang_vel_errors**2, axis=0))
    
    print("\n=== RMS Tracking Errors ===")
    print(f"Position RMS: X={rms_pos[0]:.4f}m, Y={rms_pos[1]:.4f}m, Z={rms_pos[2]:.4f}m")
    print(f"Velocity RMS: Vx={rms_vel[0]:.4f}m/s, Vy={rms_vel[1]:.4f}m/s, Vz={rms_vel[2]:.4f}m/s")
    print(f"Orientation RMS: {np.degrees(rms_angle):.4f} deg")
    print(f"Angular Vel RMS: ωx={rms_ang_vel[0]:.4f}rad/s, ωy={rms_ang_vel[1]:.4f}rad/s, ωz={rms_ang_vel[2]:.4f}rad/s")
    
    # Save figure inside the flight folder
    if cutoff_time:
        output_filename = f"error_analysis_cutoff_{cutoff_time}s.png"
    else:
        output_filename = "error_analysis.png"
    
    output_path = os.path.join(folder_path, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Error analysis plot saved to: {output_path}")
    
    plt.show()

def find_latest_flight_folder(logs_dir):
    """Find the most recent flight folder in the logs directory"""
    flight_folders = []
    
    for item in Path(logs_dir).iterdir():
        if item.is_dir() and ("_flight_" in item.name):
            flight_folders.append(item)
    
    if not flight_folders:
        return None
    
    # Sort by modification time (newest first)
    flight_folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(flight_folders[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot flight data from CoManDO MPC')
    parser.add_argument('path', nargs='?', help='Path to flight folder or logs directory')
    parser.add_argument('--cutoff', type=float, help='Cutoff time in seconds')
    parser.add_argument('--latest', action='store_true', help='Plot the latest flight in logs directory')
    parser.add_argument('--trajectory-only', action='store_true', help='Plot only trajectory analysis')
    parser.add_argument('--errors-only', action='store_true', help='Plot only error analysis')
    parser.add_argument('--all', action='store_true', help='Plot all analyses (comprehensive, trajectory, errors)')
    
    args = parser.parse_args()
    
    if args.latest:
        # Find the latest flight folder
        if args.path:
            logs_dir = args.path
        else:
            # Try to find logs directory automatically
            logs_dir = "./logs"
            if not os.path.exists(logs_dir):
                logs_dir = "../logs"  # Try one level up
        
        latest_folder = find_latest_flight_folder(logs_dir)
        if latest_folder is None:
            print(f"Error: No flight folders found in {logs_dir}")
            sys.exit(1)
        
        folder_path = latest_folder
        print(f"Using latest flight folder: {folder_path}")
    
    elif args.path:
        folder_path = args.path
        
        # Check if it's a directory
        if not os.path.isdir(folder_path):
            print(f"Error: {folder_path} is not a directory")
            print("Please provide a flight folder path")
            sys.exit(1)
    else:
        print("Usage: python3 plot_flight_logs.py [path] [options]")
        print("\nExamples:")
        print("  python3 plot_flight_logs.py /path/to/logs/cf_1_flight_20260206_182001")
        print("  python3 plot_flight_logs.py /path/to/logs/cf_1_flight_20260206_182001 --cutoff 5.0")
        print("  python3 plot_flight_logs.py --latest  # Plot the most recent flight")
        print("  python3 plot_flight_logs.py /path/to/logs --latest  # Plot latest in given directory")
        print("\nOptions:")
        print("  --cutoff TIME      : Only plot data up to TIME seconds")
        print("  --trajectory-only  : Plot only trajectory analysis")
        print("  --errors-only      : Plot only error analysis")
        print("  --all              : Plot all analyses")
        sys.exit(1)
    
    # Check if required files exist
    required_files = ["commanded_state.csv", "actual_state.csv", "control.csv"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(folder_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing files in {folder_path}:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nAvailable files:")
        for item in os.listdir(folder_path):
            print(f"  - {item}")
        sys.exit(1)
    
    print(f"Loading data from: {folder_path}")
    
    # Determine which plots to generate
    if args.trajectory_only:
        print("Generating trajectory analysis...")
        plot_3d_trajectory(folder_path, args.cutoff)
    elif args.errors_only:
        print("Generating error analysis...")
        plot_error_analysis(folder_path, args.cutoff)
    elif args.all:
        print("Generating all analyses...")
        print("\n1. Comprehensive flight data plot...")
        plot_flight_data(folder_path, args.cutoff)
        print("\n2. Trajectory & orientation analysis...")
        plot_3d_trajectory(folder_path, args.cutoff)
        print("\n3. Error analysis...")
        plot_error_analysis(folder_path, args.cutoff)
    else:
        # Default: comprehensive plot only
        print("Generating comprehensive flight data plot...")
        plot_flight_data(folder_path, args.cutoff)
        
    print("\nDone!")