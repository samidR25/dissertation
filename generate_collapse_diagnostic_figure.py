"""
generate_collapse_diagnostic_figure.py
=========================================
Illustrates what Gate 0c's collapse diagnostic actually checks, using real
LOPO window-specificity numbers plotted against the gate's real threshold
(spec_floor=0.9).

Gate 0c (collapse_diagnostic() in eval_event_level.py) FAILs a fold if ANY
of three conditions hold:
  1. window specificity < 0.9                              (over-firing)
  2. largest predicted block covers >90% of the recording   (over-firing)
  3. positive-window fraction < 10% of the true seizure rate (under-firing)

This figure visualises criterion 1 only (the one with a clean, plottable
per-fold number) -- window specificity is NOT a full explanation of every
FAIL by construction (criteria 2/3 exist to catch cases criterion 1 alone
would miss). The caption is generated dynamically from the actual data --
it states which FAIL folds ARE explained by criterion 1 alone and which
aren't, rather than a hardcoded claim that could drift out of sync with
the data.

Reads results/lopo_summary.json directly (same source as
generate_lopo_figures.py) -- no hardcoded per-patient numbers.

Usage:
    python3 generate_collapse_diagnostic_figure.py
    python3 generate_collapse_diagnostic_figure.py --summary results/lopo_summary.json --out-dir figures/
Output: collapse_diagnostic_illustration.png (300 DPI) in --out-dir
"""
import argparse, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--summary', default='results/lopo_summary.json')
parser.add_argument('--out-dir', default='figures')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

with open(args.summary) as f:
    summary = json.load(f)

rows = summary['per_fold']
DATA = {r['patient']: (r['window_specificity'], r['collapse_pass']) for r in rows}

SPEC_FLOOR = 0.9
PASS_COLOR = '#2E7D32'
FAIL_COLOR = '#C62828'

patients = sorted(DATA.keys(), key=lambda p: DATA[p][0])
specs = [DATA[p][0] for p in patients]
passed = [DATA[p][1] for p in patients]
colors = [PASS_COLOR if p else FAIL_COLOR for p in passed]

# Determine, from the actual data, which FAIL folds criterion 1 alone
# explains -- drives the caption text below dynamically instead of a
# hardcoded claim that could go stale if the data changes.
fail_patients = [p for p in patients if not DATA[p][1]]
explained_by_crit1 = [p for p in fail_patients if DATA[p][0] < SPEC_FLOOR]
not_explained = [p for p in fail_patients if DATA[p][0] >= SPEC_FLOOR]

fig, ax = plt.subplots(figsize=(7.5, 6.5))
y = np.arange(len(patients))
ax.hlines(y, 0, specs, color=colors, linewidth=2, alpha=0.6, zorder=1)
ax.scatter(specs, y, color=colors, s=90, edgecolor='black', linewidth=0.7, zorder=3)
ax.axvline(SPEC_FLOOR, color='black', linestyle='--', linewidth=1.2, zorder=2)
ax.text(SPEC_FLOOR + 0.008, len(patients) - 0.3, 'spec_floor = 0.90\n(Gate 0c criterion 1)',
        fontsize=8.5, va='top')

ax.set_yticks(y)
ax.set_yticklabels(patients)
ax.set_xlabel('Window specificity')
ax.set_xlim(min(0.3, min(specs) - 0.05), 1.03)
ax.set_title('Window specificity vs. the collapse gate\u2019s over-firing floor\n(LOPO folds)',
             fontsize=12.5)
ax.spines[['top', 'right']].set_visible(False)

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PASS_COLOR, markeredgecolor='black', markersize=9, label='Collapse PASS'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=FAIL_COLOR, markeredgecolor='black', markersize=9, label='Collapse FAIL'),
]
ax.legend(handles=legend_elems, loc='lower right', frameon=False, fontsize=9)

if not_explained:
    crit_note = (f'In this LOPO run, {len(explained_by_crit1)}/{len(fail_patients)} FAIL folds cross the\n'
                 f'specificity floor alone (criterion 1); {", ".join(not_explained)} FAIL via criteria 2/3\n'
                 f'instead (not visible in this plot) \u2014 specificity is one signal, not the whole gate.')
else:
    crit_note = (f'In this LOPO run, all {len(fail_patients)} FAIL folds happen to cross the\n'
                 'specificity floor alone (criterion 1) \u2014 criteria 2/3 exist to catch failure\n'
                 'modes criterion 1 would miss, but were not needed to explain any FAIL here.')

fig.tight_layout(rect=[0, 0.16, 1, 1])
fig.text(0.5, 0.02,
         'Gate 0c FAILs a fold if ANY of: (1) window specificity < 0.90, (2) largest predicted\n'
         'block covers >90% of the recording, or (3) positive-window fraction < 10% of the true\n'
         f'seizure rate (under-firing). {crit_note}',
         ha='center', fontsize=8, style='italic', color='#424242')

out_path = os.path.join(args.out_dir, 'collapse_diagnostic_illustration.png')
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")
