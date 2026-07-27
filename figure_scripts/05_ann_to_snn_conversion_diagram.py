import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

# Left: ReLU activation - a single bar showing continuous activation value
ax = axes[0]
act_value = 0.72
ax.bar([0], [act_value], width=0.4, color='#4C72B0')
ax.set_xlim(-1, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_ylabel('Continuous activation value')
ax.set_title('ANN: ReLU unit')
ax.text(0, act_value + 0.04, f'a = {act_value}', ha='center', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Right: LIF spike train over a short time window, rate proportional to activation
ax2 = axes[1]
T = 1.0
rate = act_value * 12  # spikes per unit window, proportional to activation
rng = np.random.default_rng(7)
n_spikes = int(round(rate))
spike_times = np.sort(rng.uniform(0, T, n_spikes))
ax2.eventplot(spike_times, lineoffsets=0.5, linelengths=0.8, colors='#55A868')
ax2.set_xlim(0, T)
ax2.set_ylim(0, 1)
ax2.set_yticks([])
ax2.set_xlabel('Time window')
ax2.set_title(f'SNN: LIF neuron ({n_spikes} spikes / window)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

fig.suptitle('Rate-coding correspondence: ReLU activation \u2192 LIF spike rate (Eq. 2.1)', fontsize=11)

# arrow between panels
fig.text(0.495, 0.5, r'$\Rightarrow$', fontsize=28, ha='center', va='center')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('./ann_to_snn_conversion_diagram.png', dpi=200)
print('done')
