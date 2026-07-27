import matplotlib.pyplot as plt
import numpy as np

# Simple LIF simulation: tau dV/dt = -V + R*I, spike at Vth, reset to Vreset
dt = 0.001
T = 0.5
n = int(T / dt)
t = np.arange(n) * dt

tau = 0.02
R = 1.0
I = 1.6          # constant input current
Vth = 1.0
Vreset = 0.0

V = np.zeros(n)
spikes = []
v = 0.0
for i in range(1, n):
    dv = dt * (-v + R * I) / tau
    v = v + dv
    if v >= Vth:
        spikes.append(t[i])
        v = Vreset
    V[i] = v

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(t * 1000, V, color='#4C72B0', linewidth=1.6, label='Membrane potential $V(t)$')
ax.axhline(Vth, color='#C44E52', linestyle='--', linewidth=1.3, label='Threshold $V_{th}$')

for s in spikes:
    ax.axvline(s * 1000, color='#55A868', linewidth=1.0, alpha=0.5, ymax=0.93)
ax.plot([], [], color='#55A868', linewidth=1.0, label='Spike event')

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane potential (a.u.)')
ax.set_title('Leaky Integrate-and-Fire neuron under constant input current\n(integrate \u2192 threshold \u2192 spike \u2192 reset, Eq. 2.1)')
ax.set_xlim(0, T * 1000)
ax.set_ylim(-0.05, 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend placed BELOW the plot, outside the data area entirely -- avoids any
# overlap with the spike lines / threshold line / curve regardless of where
# they fall.
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
          frameon=False, fontsize=9)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('./lif_membrane_potential_trace.png', dpi=200, bbox_inches='tight')
print('spikes:', len(spikes))
