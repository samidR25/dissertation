"""
generate_pipeline_diagram.py
==============================
End-to-end pipeline architecture diagram for the Methodology chapter:
CHB-MIT EEG -> preprocessing -> spatio-temporal CNN (tf-keras) ->
w4a4 quantisation -> ANN-to-SNN conversion (cnn2snn) -> AKD1000 deployment
(Raspberry Pi 5), with measured silicon results annotated at the end.

No data dependency -- purely structural, safe to regenerate any time.

Usage:
    python3 generate_pipeline_diagram.py
    python3 generate_pipeline_diagram.py --out-dir figures/

Output: pipeline_architecture.png (300 DPI, dissertation-ready) in --out-dir.
"""
import argparse, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', default='figures')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

STAGE_COLOR = '#1565C0'      # blue -- software/training stages
QUANT_COLOR = '#EF6C00'      # orange -- quantisation/conversion stages
HW_COLOR = '#2E7D32'         # green -- physical hardware
TEXT_COLOR = '#212121'
ARROW_COLOR = '#616161'

stages = [
    {
        'title': 'CHB-MIT\nScalp EEG',
        'detail': '18-ch, 256 Hz\npediatric recordings',
        'color': STAGE_COLOR,
    },
    {
        'title': 'Preprocessing',
        'detail': '512-sample windows\nuint8 scaled, chronological split',
        'color': STAGE_COLOR,
    },
    {
        'title': 'Spatio-temporal\nCNN',
        'detail': 'TensorFlow / tf-keras\n137,826 params',
        'color': STAGE_COLOR,
    },
    {
        'title': 'w4a4\nQuantisation',
        'detail': 'quantizeml\n4-bit weights, 4-bit activations',
        'color': QUANT_COLOR,
    },
    {
        'title': 'ANN\u2192SNN\nConversion',
        'detail': 'BrainChip cnn2snn\nAkidaVersion.v1',
        'color': QUANT_COLOR,
    },
    {
        'title': 'AKD1000\nDeployment',
        'detail': 'Raspberry Pi 5\nM.2 neuromorphic accelerator',
        'color': HW_COLOR,
    },
]

n = len(stages)
fig, ax = plt.subplots(figsize=(15, 4.2))

box_w, box_h = 2.0, 1.5
gap = 0.9
y_center = 0
x_positions = [i * (box_w + gap) for i in range(n)]

for i, (x, stage) in enumerate(zip(x_positions, stages)):
    box = FancyBboxPatch(
        (x, y_center - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.6, edgecolor=stage['color'], facecolor=stage['color'],
        alpha=0.15, zorder=2)
    ax.add_patch(box)
    ax.text(x + box_w / 2, y_center + 0.28, stage['title'],
            ha='center', va='center', fontsize=10.5, fontweight='bold',
            color=TEXT_COLOR, zorder=3)
    ax.text(x + box_w / 2, y_center - 0.32, stage['detail'],
            ha='center', va='center', fontsize=7.8, color='#424242',
            zorder=3, linespacing=1.5)

    if i < n - 1:
        x_next = x_positions[i + 1]
        arrow = FancyArrowPatch(
            (x + box_w, y_center), (x_next, y_center),
            arrowstyle='-|>', mutation_scale=16,
            linewidth=1.4, color=ARROW_COLOR, zorder=1)
        ax.add_patch(arrow)

# Measured silicon results, anchored under the final (hardware) stage
final_x = x_positions[-1] + box_w / 2
ax.annotate(
    '\u2248 0.906 \u00b5J / inference\n\u2248 1 mW active power\n(measured, bench DC supply)',
    xy=(final_x, y_center - box_h / 2), xytext=(final_x, y_center - 1.65),
    ha='center', va='top', fontsize=8, color=HW_COLOR, fontweight='bold',
    linespacing=1.6,
    arrowprops=dict(arrowstyle='-', color=HW_COLOR, lw=0.8, linestyle='dotted'))

# Legend distinguishing software / quantisation-conversion / hardware stages
legend_elems = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=STAGE_COLOR, alpha=0.5, markersize=12, label='Data / training (software)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=QUANT_COLOR, alpha=0.5, markersize=12, label='Quantisation / conversion'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=HW_COLOR, alpha=0.5, markersize=12, label='Physical hardware'),
]
ax.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, -0.02),
          ncol=3, frameon=False, fontsize=9)

ax.set_xlim(-0.6, x_positions[-1] + box_w + 0.6)
ax.set_ylim(-2.3, 1.2)
ax.axis('off')
ax.set_title('End-to-end pipeline: CHB-MIT EEG \u2192 CNN \u2192 quantised SNN \u2192 AKD1000',
             fontsize=13, pad=14)

fig.tight_layout()
out_path = os.path.join(args.out_dir, 'pipeline_architecture.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Saved: {out_path}")
