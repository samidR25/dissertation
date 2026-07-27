import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 4.5))

stages = [
    ("Raw scalp EEG\n(CHB-MIT)", "#8DA6CE", "§3.1"),
    ("Preprocessing +\nchronological split\n(70:15:15)", "#4C72B0", "§3.1"),
    ("seizure_cnn_v2\n(Conv\u00d73 \u2192 Dense)", "#55A868", "§3.2.1"),
    ("w4a4\nquantisation", "#DD8452", "§3.2.2"),
    ("ANN\u2192SNN\nconversion\n(cnn2snn)", "#C44E52", "§3.2.3"),
    ("AKD1000 v1\nsilicon\ndeployment", "#8172B2", "§3.5"),
]

n = len(stages)
box_w, box_h = 1.7, 1.3
xs = [i * 2.1 for i in range(n)]
y0 = 0

for i, (label, color, sec) in enumerate(stages):
    x = xs[i]
    rect = mpatches.FancyBboxPatch((x - box_w/2, y0 - box_h/2), box_w, box_h,
                                    boxstyle="round,pad=0.06", linewidth=1.2,
                                    edgecolor='black', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x, y0, label, ha='center', va='center', fontsize=9.3, color='white', weight='bold')
    ax.text(x, y0 - box_h/2 - 0.35, sec, ha='center', fontsize=10, color='#333333', weight='bold')
    if i < n - 1:
        arrow = FancyArrowPatch((x + box_w/2, y0), (xs[i+1] - box_w/2, y0),
                                 arrowstyle='-|>', mutation_scale=16, color='black', linewidth=1.4)
        ax.add_patch(arrow)

ax.set_xlim(-1.3, xs[-1] + 1.3)
ax.set_ylim(-2.0, 1.4)
ax.axis('off')
ax.set_title('End-to-end pipeline: raw EEG \u2192 quantised CNN \u2192 spiking conversion \u2192 AKD1000 v1 silicon',
             fontsize=12.5, pad=12)

plt.tight_layout()
plt.savefig('./pipeline_architecture_full.png', dpi=200)
print('done')
