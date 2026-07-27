import matplotlib.pyplot as plt
import numpy as np

patients = ['chb01','chb02','chb03','chb05','chb06','chb07','chb09',
            'chb10','chb11','chb13','chb15','chb16','chb18','chb19','chb20']

C1 = {'chb01','chb02','chb05'}
pool7 = {'chb01','chb02','chb05','chb06','chb07','chb09','chb20'}
C2 = {'chb10','chb13','chb15','chb16'}

regimes = ['C1\n(3-patient pool)', 'pool7\n(7-patient pool)', 'C2\n(fine-tuned individually)']
sets_ = [C1, pool7, C2]

fig, ax = plt.subplots(figsize=(7, 8))

for r, patient in enumerate(patients):
    y = len(patients) - r
    ax.text(-0.3, y, patient, ha='right', va='center', fontsize=9.5, family='monospace')
    for c, s in enumerate(sets_):
        x = c
        marker = '\u25CF' if patient in s else ''
        color = ['#4C72B0', '#DD8452', '#C44E52'][c]
        if marker:
            ax.text(x, y, marker, ha='center', va='center', fontsize=13, color=color)
        ax.add_patch(plt.Rectangle((x - 0.42, y - 0.42), 0.84, 0.84, fill=False,
                                    edgecolor='#dddddd', linewidth=0.6))

for c, label in enumerate(regimes):
    ax.text(c, len(patients) + 1, label, ha='center', va='bottom', fontsize=10, weight='bold')

ax.set_xlim(-2.2, 2.8)
ax.set_ylim(0, len(patients) + 2.2)
ax.axis('off')
ax.set_title('Training-regime pool composition by patient', fontsize=12, pad=6)

note = ("pool6-minus-X (Section 4.2.2): for each of pool7's seven patients, a matched\n"
        "six-patient pool excluding that one target patient, evaluated zero-shot against them.\n"
        "Not shown as a fixed column above since its membership changes per target patient.")
fig.text(0.5, 0.02, note, ha='center', fontsize=8.5, style='italic')

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig('./training_regime_pool_composition.png', dpi=200)
print('done')
