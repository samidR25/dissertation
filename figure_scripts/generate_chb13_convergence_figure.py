"""
generate_chb13_convergence_figure.py
=======================================
chb13 cross-candidate convergence: four mechanistically distinct
generalisation interventions, all evaluated against the same patient.

Data loading, in priority order:
  1. Reads live from results/ using filename patterns CONFIRMED against
     the user's actual `ls results/` output (this session, not inferred):
       DANN:   event_results_seizure_model_multi_dann_lambda{LAMBDA}_v2_w4a4_on_chb13.json
               -- only lambda0.1 and lambda0.5 exist (no lambda0.01).
               Defaults to lambda0.1; pass --dann-lambda 0.5 to use the other.
       CORAL:  event_results_seizure_model_multi_coral_lambda{LAMBDA}_v2_w4a4_on_chb13.json
               -- lambda0.01 and lambda0.1 both exist, both cited at 0.167
               in the ledger. Defaults to lambda0.01.
       SSL:    event_results_seizure_model_multi_sslpretrain_v2_w4a4_on_chb13.json
               -- single file, no lambda/seed variants.
       G:      event_results_seizure_model_multi_g_v2_w4a4_seed{42,123,2024}_on_chb13.json
               -- THREE seed files, averaged (mean +/- pstdev) to reproduce
               the 0.889 +/- 0.157 figure, not a single file read.
  2. Falls back to the last confirmed hardcoded value if a file is missing,
     with a clear warning (should only trigger if results/ has changed
     since the `ls` output this was built against).

IMPORTANT HONESTY NOTE: DANN/CORAL/SSL land at an IDENTICAL sensitivity
(0.167) in the fallback data -- a tight, genuine convergence. G does NOT
match that number -- its failure mode is a different magnitude of the same
over-firing direction. Do not let a future edit flatten this into "all
four identical" without checking the source data again.

Usage:
    python3 generate_chb13_convergence_figure.py
    python3 generate_chb13_convergence_figure.py --results-dir results --dann-lambda 0.5 --coral-lambda 0.1
Output: chb13_convergence.png (300 DPI) in --out-dir (default figures/)
"""
import argparse, glob, json, os
import statistics as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', default='results')
parser.add_argument('--out-dir', default='figures')
parser.add_argument('--dann-lambda', default='0.1', choices=['0.1', '0.5'])
parser.add_argument('--coral-lambda', default='0.01', choices=['0.01', '0.1'])
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


def load_sens(path):
    with open(path) as f:
        r = json.load(f)
    return r['event_level']['event_sensitivity']


def try_load_single(label, path, fb_sens, fb_err, note):
    full = os.path.join(args.results_dir, path)
    if os.path.exists(full):
        try:
            sens = load_sens(full)
            print(f"  {label}: loaded live sens={sens} from {full}")
            return (label, sens, None, False, note)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"WARNING: {full} exists but couldn't be parsed as expected ({e}) "
                  f"-- falling back to hardcoded value for {label}.")
    else:
        print(f"WARNING: {full} not found -- falling back to hardcoded value for "
              f"{label} ({fb_sens}). This should only happen if results/ has "
              f"changed since the `ls` output this script was built against.")
    return (label, fb_sens, fb_err, False, note)


CONDITIONS = []

# C1 baseline -- no result file, this is the pre-intervention frozen-pool number
CONDITIONS.append(('C1 baseline\n(frozen pool)', 0.833, None, False, 'zero patient data'))

CONDITIONS.append(try_load_single(
    'DANN', f'event_results_seizure_model_multi_dann_lambda{args.dann_lambda}_v2_w4a4_on_chb13.json',
    0.167, None, 'adversarial\ndomain-invariance'))

CONDITIONS.append(try_load_single(
    'CORAL', f'event_results_seizure_model_multi_coral_lambda{args.coral_lambda}_v2_w4a4_on_chb13.json',
    0.167, None, 'statistical\ncovariance alignment'))

CONDITIONS.append(try_load_single(
    'SSL-pretrain', 'event_results_seizure_model_multi_sslpretrain_v2_w4a4_on_chb13.json',
    0.167, None, 'self-supervised\ninit, no domain signal'))

# Candidate G -- multi-seed, averaged (not a single-file read)
g_seeds = [42, 123, 2024]
g_paths = [os.path.join(args.results_dir,
           f'event_results_seizure_model_multi_g_v2_w4a4_seed{s}_on_chb13.json') for s in g_seeds]
g_vals = []
for p in g_paths:
    if os.path.exists(p):
        try:
            g_vals.append(load_sens(p))
        except (KeyError, json.JSONDecodeError):
            pass
if len(g_vals) == len(g_seeds):
    g_mean, g_err = stats.mean(g_vals), stats.pstdev(g_vals)
    print(f"  Candidate G: loaded live, seeds {g_seeds} -> sens={g_vals}, "
          f"mean={g_mean:.3f} +/- {g_err:.3f}")
    CONDITIONS.append(('Candidate G\n(3-seed mean)', g_mean, g_err, False,
                        'hand-engineered\nspectral features,\nno learned repr.'))
else:
    print(f"WARNING: only found {len(g_vals)}/{len(g_seeds)} Candidate G seed files "
          f"for chb13 -- falling back to hardcoded value (0.889 +/- 0.157).")
    CONDITIONS.append(('Candidate G\n(3-seed mean)', 0.889, 0.157, False,
                        'hand-engineered\nspectral features,\nno learned repr.'))



FAIL_COLOR = '#C62828'
PASS_COLOR = '#2E7D32'

fig, ax = plt.subplots(figsize=(9.5, 5.8))
x = np.arange(len(CONDITIONS))
labels = [c[0] for c in CONDITIONS]
sens = [c[1] for c in CONDITIONS]
errs = [c[2] if c[2] is not None else 0 for c in CONDITIONS]
colors = [PASS_COLOR if c[3] else FAIL_COLOR for c in CONDITIONS]

bars = ax.bar(x, sens, yerr=errs, capsize=5, color=colors, alpha=0.85,
               edgecolor='black', linewidth=0.7, width=0.6, zorder=3)

for xi, (label, s, err, passed, note) in zip(x, CONDITIONS):
    ax.text(xi, s + (err if err else 0) + 0.03,
            f'{s:.3f}' + (f' \u00b1 {err:.3f}' if err else ''),
            ha='center', fontsize=9, fontweight='bold')
    ax.text(xi, -0.11, note, ha='center', va='top', fontsize=7.3, color='#424242',
            linespacing=1.3)
    ax.text(xi, -0.03, 'FAIL (over-firing)', ha='center', va='top', fontsize=7.8,
            color=FAIL_COLOR, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['' for _ in labels])
ax.set_ylabel('Event sensitivity (chb13)')
ax.set_ylim(-0.35, 1.45)
ax.axhline(0, color='black', linewidth=0.8)
ax.spines[['top', 'right', 'bottom']].set_visible(False)
for xi, label in zip(x, labels):
    ax.text(xi, 1.32, label, ha='center', fontsize=9.5, fontweight='bold')

ax.set_title('chb13: four mechanistically distinct interventions,\n'
             'four different sensitivity values — all collapse via over-firing',
             fontsize=12.5)

fig.tight_layout(rect=[0, 0.15, 1, 1])
fig.text(0.5, 0.02,
         'DANN, CORAL and SSL-pretrain land at an identical sensitivity (0.167) — a tight convergence.\n'
         'Candidate G differs in magnitude (0.889 \u00b1 0.157, stable across 3 seeds) but still fails the\n'
         'collapse diagnostic via the same over-firing direction. The convergence claim is about\n'
         'failure MODE, not an identical number across all four.',
         ha='center', fontsize=8.3, style='italic', color='#424242')

out_path = os.path.join(args.out_dir, 'chb13_convergence.png')
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")
