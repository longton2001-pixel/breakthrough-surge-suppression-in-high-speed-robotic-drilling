"""
Diameter Error Analysis (Fig 13)
Compares measured drilling diameter and absolute error for 5 configurations.
- Upper panel (a): Absolute diameter (mm) vs. Nominal 10.0mm.
- Lower panel (b): Diameter error (mm) for each position (A-D).
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv
import os

# ============================================================
# Global Plotting Configuration (SCI Standard)
# ============================================================
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
    'legend.fontsize': 13,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'figure.dpi': 300,
})

# Load Data
data = []
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'radius_error_data.csv')
with open(DATA_PATH, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    for row in reader:
        row = {k.strip(): v.strip() for k, v in row.items()}
        if row.get('no.', ''):
            data.append({'no': row['no.'], 'diameter': float(row['diameter'])})

groups = ['Position A', 'Position B', 'Position C', 'Position D']
methods = ['Default CAM', 'SIDSO-optimized', 'Stiffness-optimized', 'Mass-optimized', 'Random feasible']
STYLES = [
    {'color': '#a9a9a9', 'alpha': 0.65}, 
    {'color': '#d62728', 'alpha': 0.65}, 
    {'color': '#1f77b4', 'alpha': 0.65}, 
    {'color': '#2ca02c', 'alpha': 0.65}, 
    {'color': '#ff7f0e', 'alpha': 0.65}, 
]

nominal_diameter = 10.0  # mm (Target diameter)

# Extract diameters
diameters = {m: [data[g_idx * 5 + m_idx]['diameter'] for g_idx in range(4)] for m_idx, m in enumerate(methods)}
errors = {m: [d - nominal_diameter for d in diameters[m]] for m in methods}
means = {m: np.mean(diameters[m]) for m in methods}

# Plotting Layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
plt.subplots_adjust(hspace=0.1)

x = np.arange(len(groups))
bar_width = 0.15
offsets = np.array([-2, -1, 0, 1, 2]) * (bar_width + 0.02)

# --- Subplot (a): Diameter ---
ax1.axhline(y=nominal_diameter, color='#333333', linestyle='--', linewidth=1.2, alpha=0.8, label='Nominal (10.0 mm)', zorder=1)
for i, m in enumerate(methods):
    vals = diameters[m]
    sty = STYLES[i]
    bars = ax1.bar(x + offsets[i], vals, width=bar_width, label=m, color=sty['color'], alpha=sty['alpha'], edgecolor='black', linewidth=1, zorder=3)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f'{val:.2f}', ha='center', va='bottom', fontsize=9.5, color='#333333', zorder=6)
    ax1.axhline(y=means[m], color=sty['color'], linestyle='--', linewidth=1.0, alpha=0.5, zorder=2)

ax1.set_ylabel('Diameter (mm)')
ax1.yaxis.grid(True, linestyle='-', linewidth=0.3, color='#EEEEEE')
ax1.set_axisbelow(True)
ax1.legend(loc='upper right', framealpha=0.9, ncol=2, fontsize=13)
ax1.text(-0.08, 1.02, '(a)', transform=ax1.transAxes, fontsize=16, va='bottom', ha='right')

# --- Subplot (b): Diameter Error ---
for i, m in enumerate(methods):
    vals = errors[m]
    sty = STYLES[i]
    bars = ax2.bar(x + offsets[i], vals, width=bar_width, color=sty['color'], alpha=sty['alpha'], edgecolor='black', linewidth=1, zorder=3)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + (0.01 if val > 0 else -0.06), f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=11, color='#333333', zorder=6)

ax2.axhline(0, color='black', linewidth=1.5, zorder=1)
ax2.set_ylabel('Diameter Error (mm)')
ax2.set_xlabel('Drilling Position')
ax2.set_xticks(x)
ax2.set_xticklabels(groups)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.3, color='#EEEEEE')
ax2.set_axisbelow(True)
ax2.text(-0.08, 1.02, '(b)', transform=ax2.transAxes, fontsize=16, va='bottom', ha='right')

ax1.set_ylim(min([v for m in methods for v in diameters[m]])-0.8, max([v for m in methods for v in diameters[m]])+1.0)
ax2.set_ylim(min([v for m in methods for v in errors[m]])-0.3, max([v for m in methods for v in errors[m]])+0.5)

# Save
out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
if not os.path.exists(out_dir): os.makedirs(out_dir)
fig.savefig(os.path.join(out_dir, 'Fig13.png'), bbox_inches='tight', dpi=600)
fig.savefig(os.path.join(out_dir, 'Fig13.pdf'), bbox_inches='tight')
plt.close(fig)
