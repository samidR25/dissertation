"""
generate_c2_vs_lopo_figure.py
================================
Paired comparison: C2 per-patient fine-tuning (deployability question) vs.
LOPO cross-patient generalisation (field-comparability question), for the
4 patients where both canonical numbers exist (chb10/13/15/16).

Data loading, in priority order:
  1. LOPO: reads results/lopo_summary.json directly (confirmed real path,
     verified against your actual file this session).
  2. C2: reads results/event_results_seizure_model_{patient}ft_s256_v2_w4a4_on_{patient}.json
     -- the s256 seed is confirmed as the model-of-record checkpoint (matches
     the hardware_results_*_base-{patient}ft_s256.json naming seen in your
     results/ listing this session).

     IMPORTANT CAVEAT, unresolved: eval_event_level.py's --conformal flag
     applies a calibrated decision threshold but writes to the SAME output
     filename as a default-threshold run (no "conformal" tag in the
     filename). This script cannot tell from the filename alone whether
     the file on disk is the default-threshold result or the conformal
     alpha=0.01 result the ledger's canonical table (Sec.1) actually cites.
     The script prints the loaded sensitivity/FP-hr so you can manually
     compare against the ledger's confirmed values (chb10ft=1.000/5.46,
     chb13ft=0.667/2.42, chb15ft=0.571/1.33, chb16ft=0.200/9.83) -- if they
     match, you have the conformal result; if not, you likely have the
     default-threshold run and need to re-run eval_event_level.py
     --conformal --conformal-alpha 0.01 to regenerate the correct file.
  3. Falls back to the last confirmed hardcoded value if a file is missing,
     with a clear warning.

Usage:
    python3 generate_c2_vs_lopo_figure.py
    python3 generate_c2_vs_lopo_figure.py --results-dir results --summary results/lopo_summary.json
Output: c2_vs_lopo_comparison.png (300 DPI) in --out-dir (default figures/)
"""
import argparse, glob, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--summary', default='results/lopo_summary.json')
parser.add_argument('--results-dir', default='results')
parser.add_argument('--out-dir', default='figures')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

PATIENTS = ['chb10', 'chb13', 'chb15', 'chb16']

# Last verified hardcoded fallback (conformal alpha=0.01, canonical per
# Handoff_items1to3_done_item4_next.md Sec.1) -- used only if the live
# file glob below comes up empty for a given patient.
C2_FALLBACK = {
    'chb10': (1.000, 5.46),
    'chb13': (0.667, 2.42),
    'chb15': (0.571, 1.33),
    'chb16': (0.200, 9.83),
}

# ── Load LOPO (confirmed real path/schema) ──────────────────────────────
with open(args.summary) as f:
    lopo_summary = json.load(f)
lopo_rows = {r['patient']: r for r in lopo_summary['per_fold']}

# ── Load C2, with fallback + warnings ───────────────────────────────────
c2_data = {}
for p in PATIENTS:
    path = os.path.join(args.results_dir, f'event_results_seizure_model_{p}ft_s256_v2_w4a4_on_{p}.json')
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        try:
            sens = r['event_level']['event_sensitivity']
            fphr = r['event_level']['fp_per_hour']
            c2_data[p] = (sens, fphr)
            fb = C2_FALLBACK[p]
            match = abs(sens - fb[0]) < 1e-3 and abs(fphr - fb[1]) < 0.05
            print(f"  {p}: loaded live sens={sens:.3f}, fp/hr={fphr:.2f} from {path}"
                  + (" -- MATCHES ledger's conformal alpha=0.01 value, good."
                     if match else
                     f" -- does NOT match ledger's conformal value ({fb}). "
                     f"This is likely the default-threshold run, not the conformal "
                     f"result -- re-run eval_event_level.py --conformal --conformal-alpha 0.01 "
                     f"for {p} if you need the calibrated number."))
        except KeyError:
            print(f"WARNING: {path} found but missing expected event_level fields -- "
                  f"falling back to hardcoded value for {p}.")
            c2_data[p] = C2_FALLBACK[p]
    else:
        print(f"WARNING: {path} not found -- falling back to hardcoded "
              f"C2 value for {p} ({C2_FALLBACK[p]}).")
        c2_data[p] = C2_FALLBACK[p]

DATA = {
    p: (c2_data[p][0], c2_data[p][1],
        lopo_rows[p]['event_sensitivity'], lopo_rows[p]['fp_per_hour'],
        lopo_rows[p]['collapse_pass'])
    for p in PATIENTS
}


patients = list(DATA.keys())
c2_sens = [DATA[p][0] for p in patients]
lopo_sens = [DATA[p][2] for p in patients]
lopo_collapse = [DATA[p][4] for p in patients]

C2_COLOR = '#5E35B1'    # purple -- personalisation track
LOPO_PASS_COLOR = '#2E7D32'
LOPO_FAIL_COLOR = '#C62828'

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(patients))
w = 0.35

bars_c2 = ax.bar(x - w/2, c2_sens, w, color=C2_COLOR, edgecolor='black',
                  linewidth=0.6, label='C2 (per-patient fine-tune,\ndeployability)')
lopo_colors = [LOPO_PASS_COLOR if p else LOPO_FAIL_COLOR for p in lopo_collapse]
bars_lopo = ax.bar(x + w/2, lopo_sens, w, color=lopo_colors, edgecolor='black',
                    linewidth=0.6, hatch='//', label='LOPO (cross-patient,\nfield-comparability)')

for xi, v in zip(x - w/2, c2_sens):
    ax.text(xi, v + 0.02, f'{v:.3f}', ha='center', fontsize=8.5)
for xi, v, passed in zip(x + w/2, lopo_sens, lopo_collapse):
    flag = '' if passed else ' (collapse-FAIL)'
    ax.text(xi, v + 0.02, f'{v:.3f}{flag}', ha='center', fontsize=7.5,
            color='black' if passed else LOPO_FAIL_COLOR)

ax.set_xticks(x)
ax.set_xticklabels(patients)
ax.set_ylabel('Event sensitivity')
ax.set_ylim(0, 1.18)
ax.set_title('C2 vs. LOPO answer different questions —\nnot a single-number comparison',
             fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

# Footnote making the "don't over-read this" caveat visually explicit
fig.tight_layout(rect=[0, 0.14, 1, 1])
fig.text(0.5, 0.02,
         'C2 = deployable, per-patient-calibrated model. LOPO = cold cross-patient\n'
         'generalisation (no target-patient data at all). A high LOPO number is not\n'
         'required for C2 to be a valid deployment story — see Section 1e.',
         ha='center', fontsize=8, style='italic', color='#424242')

out_path = os.path.join(args.out_dir, 'c2_vs_lopo_comparison.png')
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")
