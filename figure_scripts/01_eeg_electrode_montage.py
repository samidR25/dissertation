import matplotlib.pyplot as plt
import numpy as np

# Approximate 10-20 system positions (top-down view, nose up), normalized coords
# Standard positions actually used in the project's 18-channel bipolar montage
# (derived from electrodes: FP1,FP2,F7,F3,FZ,F4,F8,T7,C3,CZ,C4,T8,P7,P3,PZ,P4,P8,O1,O2)
included_positions = {
    'FP1': (-0.35, 0.85), 'FP2': (0.35, 0.85),
    'F7': (-0.75, 0.45),  'F3': (-0.4, 0.45), 'FZ': (0.0, 0.45), 'F4': (0.4, 0.45), 'F8': (0.75, 0.45),
    'T7': (-0.95, 0.0),   'C3': (-0.4, 0.0),  'CZ': (0.0, 0.0),  'C4': (0.4, 0.0),  'T8': (0.95, 0.0),
    'P7': (-0.75, -0.45), 'P3': (-0.4, -0.45),'PZ': (0.0, -0.45),'P4': (0.4, -0.45),'P8': (0.75, -0.45),
    'O1': (-0.35, -0.85), 'O2': (0.35, -0.85),
}

# Verified against a real chb01_01.edf channel dump (23 total channels):
# indices 18-22 are P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8(duplicate) --
# five EXTRA CHANNELS, but only two of them introduce a genuinely new
# ELECTRODE position not already in the 18-channel montage: FT9 and FT10.
# (P7, T7, T8, P8 are already electrode endpoints in the included set --
# the extra channels just re-pair them differently, plus route through
# FT9/FT10.) T1/T2 do NOT appear in the real recording -- removed.
excluded_positions = {
    'FT9':  (-0.98, 0.25),
    'FT10': (0.98, 0.25),
}

fig, ax = plt.subplots(figsize=(6.5, 7))

# Head outline
head = plt.Circle((0, 0), 1.05, fill=False, linewidth=2, color='black')
ax.add_patch(head)
# nose
ax.plot([-0.08, 0, 0.08], [1.03, 1.18, 1.03], color='black', linewidth=2)
# ears
ax.add_patch(plt.Circle((-1.05, 0), 0.07, fill=False, linewidth=1.5, color='black'))
ax.add_patch(plt.Circle((1.05, 0), 0.07, fill=False, linewidth=1.5, color='black'))

for name, (x, y) in included_positions.items():
    ax.plot(x, y, 'o', markersize=15, color='#4C72B0', zorder=3)
    ax.text(x, y, name, ha='center', va='center', fontsize=6.3, color='white', weight='bold', zorder=4)

for name, (x, y) in excluded_positions.items():
    ax.plot(x, y, 'o', markersize=15, color='#dddddd', markeredgecolor='#999999', zorder=3)
    ax.text(x, y, name, ha='center', va='center', fontsize=6.3, color='#666666', zorder=4)
    # cross out
    d = 0.09
    ax.plot([x-d, x+d], [y-d, y+d], color='#C44E52', linewidth=1.8, zorder=5)
    ax.plot([x-d, x+d], [y+d, y-d], color='#C44E52', linewidth=1.8, zorder=5)

ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.35)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('10-20 electrode positions used to form the 18 common bipolar channels\n'
             '(19 electrodes in a chain montage \u2192 18 pairs, e.g. FP1-F7, F7-T7; '
             'crossed out = FT9/FT10, the only genuinely extra electrodes\n'
             'verified against a real chb01_01.edf dump -- the other 3 excluded '
             'channels re-pair existing electrodes: P7-T7, FT10-T8, T8-P8[dup])', fontsize=9.2)

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4C72B0', markersize=12, label='Included (18-channel common montage)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#dddddd', markeredgecolor='#999999', markersize=12, label='Excluded (patient-specific extras)'),
]
ax.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8.5)

plt.tight_layout()
plt.savefig('./eeg_electrode_montage.png', dpi=200, bbox_inches='tight')
print('done')
