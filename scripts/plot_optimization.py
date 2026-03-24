"""
Optimization Comparison Analysis (Fig 12)
- Reconstructs velocity amplification ratios (v_surge / v_feed) for 4 drilling positions.
- Compares 5 robotic configurations: Default, SIDSO, Stiff, Mass, and Random.
- Layout: (a) Velocity ratio bar chart, (b) Percentage change relative to Default posture.
"""

import numpy as np
import pandas as pd
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Global Plotting Configuration (SCI Standard)
# ============================================================
def apply_sci_style():
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'legend.fontsize': 13,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'figure.dpi': 300,
    })

apply_sci_style()

STYLES = [
    {'color': '#a9a9a9', 'label': 'Default CAM', 'alpha': 0.65},      
    {'color': '#d62728', 'label': 'SIDSO-optimized', 'alpha': 0.65}, 
    {'color': '#1f77b4', 'label': 'Stiffness-optimized', 'alpha': 0.65},
    {'color': '#2ca02c', 'label': 'Mass-optimized', 'alpha': 0.65},  
    {'color': '#ff7f0e', 'label': 'Random feasible', 'alpha': 0.65}, 
]

# Load Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df_path = os.path.join(BASE_DIR, 'data', 'optimization_data.csv')
df = pd.read_csv(df_path)
df.columns = df.columns.str.strip()

groups = ['A', 'B', 'C', 'D']
num_postures = len(STYLES)
v_surge = np.zeros((4, num_postures))
v_feed  = np.zeros((4, num_postures))

for gi, g in enumerate(groups):
    mask = df['no.'].str.startswith(g + '-')
    rows = df[mask].sort_values('no.').reset_index(drop=True)
    for pi in range(num_postures):
        if pi < len(rows):
            v_surge[gi, pi] = rows.iloc[pi]['vsurge']
            v_feed[gi, pi]  = rows.iloc[pi]['vfeed']

ratio = v_surge / v_feed
pct_change = np.zeros((4, num_postures))
for pi in range(num_postures):
    pct_change[:, pi] = (ratio[:, pi] - ratio[:, 0]) / ratio[:, 0] * 100

# Subplots Layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
plt.subplots_adjust(hspace=0.1)

for ax in [ax1, ax2]:
    ax.yaxis.grid(True, linestyle='-', linewidth=0.3, color='#EEEEEE')
    ax.set_axisbelow(True)

bar_width = 0.16
x = np.arange(len(groups))
offsets = [-2*bar_width, -bar_width, 0, bar_width, 2*bar_width]

# --- (a) Velocity Ratio Chart ---
for pi in range(num_postures):
    info = STYLES[pi]
    bars = ax1.bar(x + offsets[pi], ratio[:, pi], bar_width, color=info['color'], alpha=info['alpha'], edgecolor='black', linewidth=1, label=info['label'], zorder=3)
    for j, (val, bar) in enumerate(zip(ratio[:, pi], bars)):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f'{val:.2f}', ha='center', va='bottom', fontsize=11, color='#333333')
    ax1.axhline(y=np.mean(ratio[:, pi]), color=info['color'], linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)

ax1.axhline(y=1, color='#CCCCCC', linestyle='--', linewidth=1.5, zorder=1)
ax1.set_ylabel('$v_{\\mathrm{surge}}\\,/\\,v_{\\mathrm{feed}}$ Ratio')
ax1.set_ylim(0, np.max(ratio) * 1.35)
ax1.legend(loc='upper left', ncol=2, columnspacing=1.0)
ax1.text(-0.08, 1.03, '(a)', transform=ax1.transAxes, fontsize=15, va='bottom')

# --- (b) Percentage Change Chart ---
for pi in range(1, num_postures):
    info = STYLES[pi]
    bars = ax2.bar(x + offsets[pi], pct_change[:, pi], bar_width, color=info['color'], alpha=info['alpha'], edgecolor='black', linewidth=1, zorder=3)
    for j, (val, bar) in enumerate(zip(pct_change[:, pi], bars)):
        y_pos = (val+2) if val < -80 else (val - 1 if val < 0 else val + 1)
        va = 'top' if val < 0 else 'bottom'
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos, f'{val:+.0f}%', ha='center', va=va, fontsize=11, color='#333333', rotation=90)

ax2.axhline(y=0, color='black', linewidth=1.2, zorder=1)
ax2.set_ylabel('Change vs.\nBaseline (%)')
ax2.set_xlabel('Drilling Position')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Position {g}' for g in groups])
max_abs_change = np.max(np.abs(pct_change))
ax2.set_ylim(-max_abs_change * 1.4, max_abs_change * 1.4)
ax2.text(-0.08, 1.06, '(b)', transform=ax2.transAxes, fontsize=15, va='bottom')

# Export Results
out_dir = os.path.join(BASE_DIR, 'results')
if not os.path.exists(out_dir): os.makedirs(out_dir)
output_name = os.path.join(out_dir, "Fig12")
plt.savefig(output_name + ".png", bbox_inches='tight', dpi=600)
plt.savefig(output_name + ".pdf", bbox_inches='tight')
plt.close(fig)
