import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(9, 3.6))

# Track every box's extent so the axis limits can be computed from the
# actual geometry at the end, instead of guessed numbers that can clip.
extents = []  # list of (xmin, xmax, ymin, ymax)


def box(x, y, w, h, label, color, fontsize=8.3):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    linewidth=1.1, edgecolor='black', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fontsize, weight='bold')
    extents.append((x, x + w, y, y + h))
    return (x, y, w, h)


def arrow(p1, p2, label=None, color='black', label_dy=0.18):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=11, linewidth=1.1, color=color)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + label_dy, label, ha='center', fontsize=7.3, color=color)
        extents.append((mx - 0.3, mx + 0.3, my + label_dy - 0.1, my + label_dy + 0.3))


# Single inline row: bench supply -> Pi 5 -> HAT+/AKD1000
box_y = 0
box(0.0, box_y, 1.7, 1.0, 'RS PRO\nRS-3005P\nbench supply', '#C44E52')
ax.plot(1.9, box_y + 0.5, marker='o', markersize=7, color='black', zorder=5)
extents.append((1.9 - 0.1, 1.9 + 0.1, box_y + 0.5 - 0.1, box_y + 0.5 + 0.1))
ax.text(1.9, box_y + 1.15, 'measurement point\n(only metered path)', ha='center', fontsize=6.6, style='italic')
extents.append((1.9 - 0.8, 1.9 + 0.8, box_y + 1.15 - 0.1, box_y + 1.4))

arrow((2.0, box_y + 0.5), (2.9, box_y + 0.5), '5.00V USB-C', label_dy=0.16)

box(2.9, box_y - 0.15, 1.9, 1.3, 'Raspberry Pi 5\nSoC + RAM\n(5V input rail)', '#4C72B0')

arrow((4.8, box_y + 0.5), (5.7, box_y + 0.5), 'internal 3.3V', label_dy=0.16)

box(5.7, box_y, 1.9, 1.0, 'M.2 HAT+ \u2192\nAKD1000 v1', '#DD8452')

# Fan as a small compact branch below the Pi 5 box only (not extending overall width)
arrow((3.85, box_y - 0.15), (3.85, box_y - 0.85))
box(2.95, box_y - 1.55, 1.8, 0.7, 'GPIO fan\n(5V/0.12A, const.)', '#8DA6CE', fontsize=7.2)

# Caption, centred under the whole diagram, wrapped to stay within the row width
caption = ('No separately-metered path exists to the AKD1000 on a standard M.2 HAT+ --\n'
           'the bench measurement is necessarily system-level, not chip-isolated.')
ax.text(3.75, box_y - 2.15, caption, ha='center', fontsize=7.6, style='italic')
extents.append((3.75 - 3.6, 3.75 + 3.6, box_y - 2.35, box_y - 1.9))

# Compute bounds dynamically from every element actually drawn, plus a small margin
xs = [e[0] for e in extents] + [e[1] for e in extents]
ys = [e[2] for e in extents] + [e[3] for e in extents]
margin = 0.25
ax.set_xlim(min(xs) - margin, max(xs) + margin)
ax.set_ylim(min(ys) - margin, max(ys) + margin)

ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Power delivery path: why the bench measurement is system-level', fontsize=10.5, pad=6)

plt.tight_layout(pad=0.6)
plt.savefig('./power_measurement_circuit_diagram.png', dpi=220,
            bbox_inches='tight', pad_inches=0.15)
print('done')
