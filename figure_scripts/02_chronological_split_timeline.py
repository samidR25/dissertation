import matplotlib.pyplot as plt

patients = ['chb01','chb02','chb03','chb05','chb06','chb07','chb09',
            'chb10','chb11','chb13','chb15','chb16','chb18','chb19','chb20']

train_frac, val_frac, test_frac = 0.70, 0.15, 0.15

fig, ax = plt.subplots(figsize=(8, 6))

for i, p in enumerate(patients):
    y = len(patients) - i
    ax.barh(y, train_frac, left=0, color='#4C72B0', height=0.6, label='Train (70%)' if i == 0 else None)
    ax.barh(y, val_frac, left=train_frac, color='#DD8452', height=0.6, label='Validation (15%)' if i == 0 else None)
    ax.barh(y, test_frac, left=train_frac + val_frac, color='#C44E52', height=0.6, label='Test (15%)' if i == 0 else None)

ax.set_yticks(range(1, len(patients) + 1))
ax.set_yticklabels(patients[::-1], family='monospace', fontsize=9)
ax.set_xlim(0, 1)
ax.set_xlabel('Fraction of each patient\'s recording, in chronological order \u2192')
ax.set_title('Chronological 70:15:15 train / validation / test split per patient')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('./chronological_split_timeline.png', dpi=200, bbox_inches='tight')
print('done')
