"""
generate_energy_tradeoff_figure.py
=====================================
Illustrates the triage-device framing (D-dimer/PE analogy) with real
numbers: energy-per-inference vs. event sensitivity, comparing this
project's AKD1000-measured figure against published low-power/embedded
seizure-detection systems.

HONESTY CAVEAT (keep this annotation in the dissertation version): the
comparator systems below were evaluated on DIFFERENT datasets (PEDESITE,
not CHB-MIT) with different eval protocols and sensitivity definitions
(sample-based vs event-based vary by source). This is an illustrative
comparison of the broad energy/accuracy design space this project sits
in, NOT a strict apples-to-apples benchmark claim. State this plainly
near the figure, the same way the Ali et al. LOPO comparison caveat is
stated.

Sources (paraphrased from search, 11 July 2026 -- verify against the
actual papers before final submission, these are secondary paraphrases):
  - BrainFuseNet (Casson et al./ETH Zurich group, GAP9 PULP MCU):
    ~0.11 mJ/inference (110 uJ), EEG-only sample sensitivity ~60.7%,
    ~1.18 FP/hr, PEDESITE dataset.
  - EpiDeNet (same PEDESITE lineage): <10 mJ/inference (10,000 uJ) for
    scalp EEG, <0.5 mJ/inference (500 uJ) for intracranial EEG.
  - Xylo (SynSense SNN edge chip, sub-mW class): ~375 uW average power
    (87.4 uW IO + 287.9 uW compute) -- POWER not energy/inference, not
    directly plotted here (different unit, would need inference duration
    to convert -- noted as text only, not on the same axis).
  - UltraTrail accelerator (CHB-MIT, TC-ResNet, 4-bit fixed point):
    ~495 nW average power, ~92.3% accuracy (not directly comparable
    sensitivity metric) -- also POWER not energy/inference, text only.

REQUIRED BEFORE USING IN THE DISSERTATION:
  1. Verify each comparator number against the actual source paper --
     these were extracted from search-result snippets, not read in full.
  2. Fill in YOUR_C2_SENSITIVITY below with your own verified aggregate
     C2 (per-patient fine-tuned) sensitivity figure. This script does
     NOT invent that number -- only the two already-confirmed LOPO
     figures (PASS-only mean, and chb10's clean best-case) are
     pre-filled. Do not submit this figure with the placeholder still in.

Usage:
    python3 generate_energy_tradeoff_figure.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('figures', exist_ok=True)

# ── This project's own measured/computed figures ────────────────────────────
OWN_ENERGY_UJ = 0.906  # measured, AKD1000 silicon, w4a4 seizure_cnn_v2

# CONFIRMED from this session's LOPO sweep (results/lopo_summary.json):
LOPO_PASS_ONLY_MEAN_SENS = 0.470   # PASS-only grand mean, 8/15 folds
LOPO_BEST_CLEAN_FOLD_SENS = 1.000  # chb10 -- collapse PASS, best clean single fold

# NOT YET FILLED IN -- replace with your own verified C2 aggregate figure
# before this goes in the dissertation. Leaving as None so the script
# visibly skips it rather than silently plotting a fabricated number.
YOUR_C2_SENSITIVITY = None  # e.g. 0.83 -- fill in once you have the real aggregate

# ── Published comparators (paraphrased from search, verify before submission) ─
comparators = [
    {'name': 'BrainFuseNet\n(GAP9, EEG-only)', 'energy_uj': 110, 'sens': 0.607, 'dataset': 'PEDESITE'},
    {'name': 'EpiDeNet\n(scalp EEG)', 'energy_uj': 10000, 'sens': 0.607, 'dataset': 'PEDESITE'},
    {'name': 'EpiDeNet\n(intracranial)', 'energy_uj': 500, 'sens': 0.607, 'dataset': 'PEDESITE'},
]

fig, ax = plt.subplots(figsize=(8, 5.5))

for c in comparators:
    ax.scatter(c['energy_uj'], c['sens'], s=110, color='#9E9E9E', edgecolors='black',
               zorder=3, marker='s')
    ax.annotate(c['name'], (c['energy_uj'], c['sens']), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color='#555555')

ax.scatter(OWN_ENERGY_UJ, LOPO_PASS_ONLY_MEAN_SENS, s=160, color='#C62828',
           edgecolors='black', zorder=4, marker='o')
ax.annotate('This project\n(LOPO, PASS-only mean,\ncross-patient, CHB-MIT)',
            (OWN_ENERGY_UJ, LOPO_PASS_ONLY_MEAN_SENS), textcoords="offset points",
            xytext=(10, -28), fontsize=8, color='#C62828', fontweight='bold')

ax.scatter(OWN_ENERGY_UJ, LOPO_BEST_CLEAN_FOLD_SENS, s=160, color='#2E7D32',
           edgecolors='black', zorder=4, marker='o')
ax.annotate('This project\n(best clean fold, chb10)',
            (OWN_ENERGY_UJ, LOPO_BEST_CLEAN_FOLD_SENS), textcoords="offset points",
            xytext=(10, 6), fontsize=8, color='#2E7D32', fontweight='bold')

if YOUR_C2_SENSITIVITY is not None:
    ax.scatter(OWN_ENERGY_UJ, YOUR_C2_SENSITIVITY, s=160, color='#1565C0',
               edgecolors='black', zorder=4, marker='D')
    ax.annotate('This project\n(C2, personalised)',
                (OWN_ENERGY_UJ, YOUR_C2_SENSITIVITY), textcoords="offset points",
                xytext=(10, 6), fontsize=8, color='#1565C0', fontweight='bold')
else:
    print("NOTE: YOUR_C2_SENSITIVITY is not set -- the C2 point was skipped, "
          "not plotted as a placeholder. Fill it in and re-run before using "
          "this figure in the dissertation.")

ax.set_xscale('log')
ax.set_xlabel('Energy per inference (µJ, log scale)')
ax.set_ylabel('Event / sample sensitivity (see caption -- definitions vary by source)')
ax.set_ylim(0, 1.15)
ax.set_title('Energy vs. sensitivity: where a triage-tier device should sit\n'
              '(illustrative — different datasets/protocols, see caption before citing)',
              fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig('figures/energy_vs_sensitivity_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("Saved: figures/energy_vs_sensitivity_tradeoff.png")
print()
print("Power (not energy/inference — different unit, NOT on this plot; use as text only):")
print("  This project (measured, active)  : ~1 mW")
print("  Xylo (SynSense SNN chip)          : ~375 uW (87.4 IO + 287.9 compute)")
print("  UltraTrail accelerator            : ~495 nW average")
print("  -> Different measurement conventions (active vs. average-across-duty-cycle) —")
print("     do not present these as directly comparable without checking each")
print("     source's methodology first.")
