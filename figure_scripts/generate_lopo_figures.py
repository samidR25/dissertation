"""
generate_lopo_figures.py
===========================
Generates the three LOPO figures scoped in the write-up handoff:
  1. Bimodal sensitivity histogram (collapse-PASS folds only)
  2. Sensitivity-vs-FP/hr scatter, coloured by collapse PASS/FAIL
  3. Per-patient bar chart (event sensitivity + FP/hr, twin axes)

Reads results/lopo_summary.json directly (produced by
src/evaluation/aggregate_lopo_results.py) -- no other inputs needed.

Usage:
    python3 generate_lopo_figures.py
    python3 generate_lopo_figures.py --summary results/lopo_summary.json --out-dir figures/

Output: three .png files (300 DPI, dissertation-ready) in --out-dir.
"""
import argparse, json, os
import matplotlib
matplotlib.use('Agg')  # no display needed, safe for headless/WSL2
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--summary', default='results/lopo_summary.json')
parser.add_argument('--out-dir', default='figures')
args = parser.parse_args()

with open(args.summary) as f:
    summary = json.load(f)

rows = sorted(summary['per_fold'], key=lambda r: r['patient'])
os.makedirs(args.out_dir, exist_ok=True)

PASS_COLOR = '#2E7D32'   # green
FAIL_COLOR = '#C62828'   # red

# ── Figure 1: bimodal sensitivity histogram (collapse-PASS folds only) ──────
pass_sens = [r['event_sensitivity'] for r in rows if r['collapse_pass'] and r['event_sensitivity'] is not None]

fig, ax = plt.subplots(figsize=(6, 4))
bins = np.arange(0, 1.11, 0.1)
ax.hist(pass_sens, bins=bins, color=PASS_COLOR, edgecolor='black', alpha=0.85)
ax.set_xlabel('Event sensitivity')
ax.set_ylabel('Number of patients (collapse-PASS folds)')
ax.set_title('LOPO event sensitivity is bimodal, not a gradient\n'
              f'(n={len(pass_sens)} collapse-PASS folds of {len(rows)} total)')
ax.set_xticks(np.arange(0, 1.1, 0.2))
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(args.out_dir, 'lopo_sensitivity_bimodal_hist.png'), dpi=300)
plt.close(fig)

# ── Figure 2: sensitivity vs FP/hr scatter, coloured by collapse PASS/FAIL ──
fig, ax = plt.subplots(figsize=(6.5, 5))
texts = []
for r in rows:
    if r['event_sensitivity'] is None or r['fp_per_hour'] is None:
        continue
    color = PASS_COLOR if r['collapse_pass'] else FAIL_COLOR
    marker = 'o' if r['collapse_pass'] else 'x'
    ax.scatter(r['fp_per_hour'], r['event_sensitivity'], c=color, marker=marker,
               s=90, edgecolors='black' if r['collapse_pass'] else None, linewidths=0.8, zorder=3)
    texts.append(ax.text(r['fp_per_hour'], r['event_sensitivity'], r['patient'].replace('chb', ''),
                          fontsize=8, zorder=4))

# Several points sit close together at low FP/hr and at sensitivity==1.0
# (chb05/chb18, chb13/chb07, chb06/chb03, chb11/chb10) -- plain fixed-offset
# annotate() renders these as unreadable merged strings. adjustText nudges
# overlapping labels apart automatically and draws a thin leader line back
# to the true point so the association stays unambiguous.
try:
    from adjustText import adjust_text
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7),
                expand_points=(1.4, 1.6), force_points=(0.3, 0.4))
except ImportError:
    raise ImportError(
        "adjustText is required to keep close-together point labels readable "
        "(pip install adjustText --break-system-packages). Falling back to "
        "overlapping fixed-offset labels would reintroduce the collision bug "
        "this patch fixes.")
ax.set_xlabel('False positives / hour')
ax.set_ylabel('Event sensitivity')
ax.set_title('Perfect sensitivity is often the collapse artefact, not detection')
ax.set_ylim(-0.05, 1.08)
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PASS_COLOR, markeredgecolor='black', markersize=9, label='Collapse PASS'),
    Line2D([0], [0], marker='x', color=FAIL_COLOR, markersize=9, label='Collapse FAIL', linestyle='None'),
]
ax.legend(handles=legend_elems, loc='center right', bbox_to_anchor=(1.32, 0.5), frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.subplots_adjust(right=0.78)
fig.savefig(os.path.join(args.out_dir, 'lopo_sensitivity_vs_fphr_scatter.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ── Figure 3: per-patient bar chart, twin axes, collapse-FAIL flagged ───────
patients = [r['patient'] for r in rows]
sens = [r['event_sensitivity'] if r['event_sensitivity'] is not None else 0 for r in rows]
fphr = [r['fp_per_hour'] if r['fp_per_hour'] is not None else 0 for r in rows]
pass_flags = [r['collapse_pass'] for r in rows]

fig, ax1 = plt.subplots(figsize=(11, 5))
x = np.arange(len(patients))
bar_colors = [PASS_COLOR if p else FAIL_COLOR for p in pass_flags]
bars = ax1.bar(x, sens, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.6, label='Event sensitivity')
ax1.set_ylabel('Event sensitivity')
ax1.set_ylim(0, 1.15)
ax1.set_xticks(x)
ax1.set_xticklabels(patients, rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.plot(x, fphr, color='#1565C0', marker='D', markersize=5, linewidth=1.4, label='FP/hr')
ax2.set_ylabel('False positives / hour', color='#1565C0')
ax2.tick_params(axis='y', labelcolor='#1565C0')

for xi, p in zip(x, pass_flags):
    if not p:
        ax1.text(xi, 1.06, 'FAIL', ha='center', fontsize=7, color=FAIL_COLOR, fontweight='bold')

ax1.set_title('LOPO per-patient results — collapse-FAIL folds flagged '
              f'({sum(not p for p in pass_flags)}/{len(pass_flags)})')
legend_elems2 = [
    Line2D([0], [0], color=PASS_COLOR, lw=8, label='Collapse PASS (sensitivity)'),
    Line2D([0], [0], color=FAIL_COLOR, lw=8, label='Collapse FAIL (sensitivity)'),
    Line2D([0], [0], color='#1565C0', marker='D', label='FP/hr'),
]
ax1.legend(handles=legend_elems2, loc='upper left', frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(args.out_dir, 'lopo_per_patient_bars.png'), dpi=300)
plt.close(fig)

print(f"Saved 3 figures to {args.out_dir}/:")
print("  lopo_sensitivity_bimodal_hist.png")
print("  lopo_sensitivity_vs_fphr_scatter.png")
print("  lopo_per_patient_bars.png")
