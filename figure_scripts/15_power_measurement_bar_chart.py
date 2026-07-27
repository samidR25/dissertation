import matplotlib.pyplot as plt
import numpy as np

trials = ['Trial 1', 'Trial 2', 'Trial 3']
watts = [5.57, 5.68, 5.63]
mean_w = np.mean(watts)

fig, ax = plt.subplots(figsize=(6, 4.2))
colors = ['#4C72B0', '#4C72B0', '#4C72B0']
bars = ax.bar(trials, watts, color=colors, width=0.55, zorder=3)

ax.axhline(mean_w, color='#C44E52', linestyle='--', linewidth=1.6, zorder=2,
           label=f'Mean = {mean_w:.2f} W')

for b, w in zip(bars, watts):
    ax.text(b.get_x() + b.get_width() / 2, w + 0.03, f'{w:.2f} W',
            ha='center', va='bottom', fontsize=10)

ax.set_ylabel('System-level incremental power (W)')
ax.set_title('Bench power measurement: idle-to-active delta\n(RS PRO RS-3005P, 5.00 V, three trials)')
ax.set_ylim(0, 6.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
ax.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('./power_measurement_bar_chart.png', dpi=200)
print('done')
