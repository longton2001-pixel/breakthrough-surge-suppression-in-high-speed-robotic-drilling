"""
Experimental Validation Analysis (Fig 9)
Comparison between measured and theoretical surge velocity.
- Upper panel: Line chart for velocity comparison.
- Lower panel (background): Relative error bars with mean error reference line.
"""

import numpy as np
import pandas as pd
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ============================================================
# Global Style Configuration (SCI Standard)
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

# Load Data
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "validation_data.csv")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

v_measured = df['v_surge'].values
v_theory   = df['v_surge_theory'].values
sample_ids = np.arange(1, len(v_measured) + 1)
relative_error = np.abs(v_measured - v_theory) / v_measured * 100
mean_rel_error = np.mean(relative_error)

# Color Palette
C_MEAS, C_THEO, C_ERR, C_MEAN = '#1F77B4', '#D62728', '#808080', '#505050'

fig, ax1 = plt.subplots(figsize=(6.5, 3.8))

# --- Relative Error (Background Bars) ---
ax2 = ax1.twinx()
ax2.bar(sample_ids, relative_error, width=0.45, color=C_ERR, alpha=0.50, label='Relative error', zorder=1)
ax2.axhline(y=mean_rel_error, color=C_MEAN, linestyle='--', linewidth=1.2, label=f'Mean error ({mean_rel_error:.1f}%)', zorder=2)
ax2.set_ylim(0, max(relative_error) * 4.0)
ax2.set_ylabel('Relative error (%)', color='#333333')
ax2.tick_params(axis='y', colors='#333333')

# --- Surge Velocity (Foreground Lines) ---
x_interp = np.linspace(1, len(v_measured), 200)
y_meas_interp = np.interp(x_interp, sample_ids, v_measured)
y_theo_interp = np.interp(x_interp, sample_ids, v_theory)

ax1.fill_between(x_interp, y_meas_interp, y_theo_interp, color='#E4EDF4', alpha=0.55, zorder=2)
ax1.plot(sample_ids, v_measured, '-o', color=C_MEAS, markersize=6, markerfacecolor='white', markeredgewidth=1.5, linewidth=1.5, label='Measured $v_{\\mathrm{surge}}$', zorder=4)
ax1.plot(sample_ids, v_theory, '--s', color=C_THEO, markersize=5, markerfacecolor='white', markeredgewidth=1.2, linewidth=1.2, label='Predicted $v_{\\mathrm{surge}}$', zorder=4)

ax1.set_xlabel('Experiment Group ID')
ax1.set_ylabel('Surge Velocity (mm/s)')
ax1.set_xticks(sample_ids)
ax1.set_xlim(0.4, len(v_measured) + 0.6)
ax1.set_ylim(0, max(max(v_measured), max(v_theory)) * 1.20)
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc='upper left', bbox_to_anchor=(0.02, 0.98))

# Save Results
out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
if not os.path.exists(out_dir): os.makedirs(out_dir)
fig.savefig(os.path.join(out_dir, 'Fig9.png'))
fig.savefig(os.path.join(out_dir, 'Fig9.pdf'))
plt.close(fig)
