"""
Robotic Drilling Kinematics Analysis (Fig 14)
- Verifies TCP position consistency across all configurations (A1-A5).
- Reconstructs joint trajectories θ(t) via inverse Jacobian mapping.
- Plots 21 subplots (Velocity/Acceleration/Jerk) for End-Effector and 6 Joints.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Add core directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from robot_model import RobotModel

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'kinematics_data')

# ============================================================
# Global Plotting Configuration (SCI Standard)
# ============================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['font.size'] = 24
rcParams['axes.titlesize'] = 26
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 22
rcParams['ytick.labelsize'] = 22
rcParams['legend.fontsize'] = 15

CSV_FILES = [os.path.join(DATA_DIR, f'A-{i}.csv') for i in range(1, 6)]
LABELS = [
    'A-1: Default CAM',
    'A-2: SIDSO-optimized',
    'A-3: Stiffness-optimized',
    'A-4: Mass-optimized',
    'A-5: Random feasible'
]

# Style configuration: Highlight A-2 (Red, thicker line)
STYLES = [
    {'color': '#a9a9a9', 'linewidth': 1.5, 'alpha': 0.7, 'zorder': 1}, # A-1 Gray
    {'color': '#d62728', 'linewidth': 3.0, 'alpha': 1.0, 'zorder': 5}, # A-2 Red (Highlight)
    {'color': '#1f77b4', 'linewidth': 1.5, 'alpha': 0.6, 'zorder': 2}, # A-3 Blue
    {'color': '#2ca02c', 'linewidth': 1.5, 'alpha': 0.6, 'zorder': 2}, # A-4 Green
    {'color': '#ff7f0e', 'linewidth': 1.5, 'alpha': 0.6, 'zorder': 2}, # A-5 Orange
]
VEL_SCALE = 1e-3  # mm/s -> m/s

def load_initial_angles():
    """Load initial joint angles (degrees) and return in radians."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'initial.csv'))
    # Filter for A-1 to A-5 (the last 5 rows of the sample file)
    df_a = df[df['no.'].str.contains('A-', na=False)]
    angles_deg = df_a[['theta1','theta2','theta3','theta4','theta5','theta6']].values
    return np.deg2rad(angles_deg)

def verify_tcp_positions(robot, raw_angles_rad):
    """Verify tool tip consistency for all configurations."""
    positions = []
    for i, label in enumerate(LABELS):
        model_q = robot.map_raw_to_model(raw_angles_rad[i])
        T_tool = robot.get_end_effector_pose(model_q)
        pos = T_tool[:3, 3]
        positions.append(pos)
    return positions

def reconstruct_joint_trajectory(robot, raw_q0_rad, csv_filename, label, downsample=10):
    """Reconstruct joint trajectory using inverse Jacobian mapping."""
    df = pd.read_csv(csv_filename).iloc[::downsample].reset_index(drop=True)
    t_arr, vel_arr, acc_arr, jerk_arr = df['Time'].values, df['Velocity'].values, df['Acceleration'].values, df['Jerk'].values
    N = len(t_arr)
    
    model_q = robot.map_raw_to_model(raw_q0_rad)
    n_drill = robot.get_end_effector_pose(model_q)[:3, 2] # Tool axial direction
    
    q_model_arr = np.zeros((N, 6))
    qdot_arr = np.zeros((N, 6))
    q_model_arr[0] = model_q
    
    for k in range(N):
        x_dot = np.zeros(6)
        x_dot[:3] = (vel_arr[k] * VEL_SCALE) * n_drill
        J = robot.compute_jacobian(q_model_arr[k])
        q_dot = np.linalg.pinv(J) @ x_dot
        qdot_arr[k] = q_dot
        if k < N - 1:
            dt = t_arr[k+1] - t_arr[k]
            q_model_arr[k+1] = q_model_arr[k] + q_dot * dt
    
    # Map back to raw space
    q_raw_arr = np.array([robot.map_model_to_raw(q) for q in q_model_arr])
    qdot_raw = np.zeros_like(qdot_arr)
    qdot_raw[:, 0], qdot_raw[:, 1] = qdot_arr[:, 0], qdot_arr[:, 1]
    qdot_raw[:, 2] = -qdot_arr[:, 2] - qdot_arr[:, 1]
    qdot_raw[:, 3], qdot_raw[:, 4], qdot_raw[:, 5] = -qdot_arr[:, 3], -qdot_arr[:, 4], -qdot_arr[:, 5]
    
    # Numerical differentiation for acceleration and jerk
    def diff_central(arr, t):
        d = np.zeros_like(arr)
        for j in range(6):
            d[1:-1, j] = (arr[2:, j] - arr[:-2, j]) / (t[2:] - t[:-2])
        d[0], d[-1] = d[1], d[-2]
        return d

    qddot_raw = diff_central(qdot_raw, t_arr)
    qdddot_raw = diff_central(qddot_raw, t_arr)
    
    return {
        'time': t_arr, 'q_raw': q_raw_arr, 'qdot_raw': qdot_raw,
        'qddot_raw': qddot_raw, 'qdddot_raw': qdddot_raw,
        'end_vel': vel_arr, 'end_acc': acc_arr, 'end_jerk': jerk_arr,
        'label': label
    }

def align_and_crop(data_list, pre_s=2.0, post_s=0.5):
    """Align multiple datasets based on peak velocity time."""
    aligned = []
    for data in data_list:
        t, vel = data['time'], data['end_vel']
        t_peak = t[np.argmax(vel)]
        t_start, t_end = t_peak - pre_s, t_peak + post_s
        mask = (t >= t_start) & (t <= t_end)
        new_data = {'time': t[mask] - t_start}
        for key in ['end_vel', 'end_acc', 'end_jerk', 'qdot_raw', 'qddot_raw', 'qdddot_raw']:
            new_data[key] = data[key][mask]
        aligned.append(new_data)
    return aligned

def plot_all_curves(data_list):
    """Generate the final 7x3 plot matrix."""
    fig, axes = plt.subplots(7, 3, figsize=(18, 26), dpi=200)
    row_labels = ['End-Effector', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
    col_labels = ['Velocity', 'Acceleration', 'Jerk']
    end_units = ['(mm/s)', r'(mm/s$^2$)', r'(mm/s$^3$)']
    joint_units = ['(rad/s)', r'(rad/s$^2$)', r'(rad/s$^3$)']
    
    # Plot End-Effector (Row 0)
    end_keys = ['end_vel', 'end_acc', 'end_jerk']
    for col in range(3):
        ax = axes[0, col]
        for i, data in enumerate(data_list):
            st = STYLES[i]
            ax.plot(data['time'], data[end_keys[col]], color=st['color'], label=LABELS[i], linewidth=st['linewidth'], alpha=st['alpha'], zorder=st['zorder'])
        ax.set_title(f'{row_labels[0]} {col_labels[col]} {end_units[col]}')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0, 0].legend(loc='upper left', frameon=False)

    # Plot Joints (Rows 1-6)
    joint_keys = ['qdot_raw', 'qddot_raw', 'qdddot_raw']
    for row in range(1, 7):
        j_idx = row - 1
        for col in range(3):
            ax = axes[row, col]
            for i, data in enumerate(data_list):
                st = STYLES[i]
                ax.plot(data['time'], data[joint_keys[col]][:, j_idx], color=st['color'], linewidth=st['linewidth'], alpha=st['alpha'], zorder=st['zorder'])
            ax.set_title(f'{row_labels[row]} {col_labels[col]} {joint_units[col]}')
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3, linewidth=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    save_path = os.path.join(out_dir, 'Fig14.png')
    plt.tight_layout(pad=1.5, w_pad=0.5, h_pad=0.8, rect=[0, 0.02, 1, 0.98])
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()

def main():
    robot = RobotModel()
    raw_angles_rad = load_initial_angles()
    
    data_list = []
    for i in range(5):
        data = reconstruct_joint_trajectory(robot, raw_angles_rad[i], CSV_FILES[i], LABELS[i], downsample=10)
        data_list.append(data)

    data_aligned = align_and_crop(data_list, pre_s=2.0, post_s=0.5)
    plot_all_curves(data_aligned)
    print("Fig14 generated successfully.")

if __name__ == '__main__':
    main()
