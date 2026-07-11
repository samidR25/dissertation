"""
src/evaluation/aggregate_lopo_results.py
==========================================
Aggregates every LOPO fold's event_results_*.json (produced by
eval_event_level.py --lopo-full) into one summary table + grand mean,
reported against Ali et al. (2024)'s 72-75% true-LOPO benchmark.

Full metric bundle per fold, wins and losses both -- never sensitivity
alone, per standing project discipline. Every fold appears in the table
even if its result is a known-expected negative (e.g. chb03).

Usage:
    python3 src/evaluation/aggregate_lopo_results.py
    python3 src/evaluation/aggregate_lopo_results.py --out results/lopo_summary.json
"""
import argparse, glob, json, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', default='results')
parser.add_argument('--out', default='results/lopo_summary.json')
args = parser.parse_args()

# Only pick up files from a --lopo-full run against a multi_lopo_* fold model
# -- avoids accidentally sweeping up unrelated event_results_*.json files
# from earlier (non-LOPO) sessions sitting in the same results/ directory.
pattern = os.path.join(args.results_dir, 'event_results_seizure_model_multi_lopo_*_on_*.json')
paths = sorted(glob.glob(pattern))

if not paths:
    raise SystemExit(
        f"No LOPO result files found matching {pattern}.\n"
        "Run the sweep first: bash run_lopo_sweep.sh")

rows = []
for p in paths:
    with open(p) as f:
        r = json.load(f)
    ev = r['event_level']
    rows.append({
        'patient':            r['eval_patient'],
        'event_sensitivity':  ev['event_sensitivity'],
        'fp_per_hour':        ev['fp_per_hour'],
        'window_specificity': r['window_level']['specificity'],
        'window_auprc':       r['window_level']['auprc'],
        'collapse_pass':      r['collapse_diagnostic']['pass'],
        'n_events':           ev['n_events'],
        'n_detected':         ev['n_detected'],
        'total_hours':        ev['total_hours'],
        'source_file':        os.path.basename(p),
    })

rows.sort(key=lambda x: x['patient'])

print(f"{'Patient':<8} {'EvtSens':>8} {'FP/hr':>8} {'WinSpec':>8} "
      f"{'AUPRC':>7} {'Events':>10} {'Hours':>7} {'Collapse':>9}")
print('-' * 70)
for r in rows:
    es  = f"{r['event_sensitivity']:.3f}" if r['event_sensitivity'] is not None else "  N/A"
    fpr = f"{r['fp_per_hour']:.2f}" if r['fp_per_hour'] is not None else "  N/A"
    ws  = f"{r['window_specificity']:.3f}" if r['window_specificity'] is not None else "  N/A"
    ap  = f"{r['window_auprc']:.3f}" if r['window_auprc'] is not None else "  N/A"
    col = "PASS" if r['collapse_pass'] else "FAIL"
    print(f"{r['patient']:<8} {es:>8} {fpr:>8} {ws:>8} {ap:>7} "
          f"{r['n_detected']:>4}/{r['n_events']:<4} {r['total_hours']:>7.1f} {col:>9}")

sens_vals = [r['event_sensitivity'] for r in rows if r['event_sensitivity'] is not None]
fpr_vals  = [r['fp_per_hour']       for r in rows if r['fp_per_hour']       is not None]
n_pass    = sum(1 for r in rows if r['collapse_pass'])

# Collapse-PASS-only figures, separate from the all-folds figures below.
# A collapse-FAIL fold's sensitivity is not a real detection result -- e.g.
# a model that predicts positive for ~100% of the recording will show a
# misleadingly perfect event sensitivity (every real seizure overlaps the
# one giant predicted block) while genuinely detecting nothing. Averaging
# that in with real results silently inflates the headline number. Report
# both, but the PASS-only figure is the one that belongs in a results
# table without a caveat attached.
pass_sens_vals = [r['event_sensitivity'] for r in rows if r['event_sensitivity'] is not None and r['collapse_pass']]
pass_fpr_vals  = [r['fp_per_hour']       for r in rows if r['fp_per_hour']       is not None and r['collapse_pass']]

grand_mean_sens = float(np.mean(sens_vals)) if sens_vals else None
grand_std_sens  = float(np.std(sens_vals))  if sens_vals else None
grand_mean_fpr  = float(np.mean(fpr_vals))  if fpr_vals  else None

pass_mean_sens = float(np.mean(pass_sens_vals)) if pass_sens_vals else None
pass_std_sens  = float(np.std(pass_sens_vals))  if pass_sens_vals else None
pass_mean_fpr  = float(np.mean(pass_fpr_vals))  if pass_fpr_vals  else None

print('-' * 70)
print(f"Folds run        : {len(rows)}")
print(f"Collapse PASS     : {n_pass}/{len(rows)}")
if grand_mean_sens is not None:
    print(f"Grand mean event sensitivity (ALL folds)        : {grand_mean_sens:.3f} ± {grand_std_sens:.3f}")
if grand_mean_fpr is not None:
    print(f"Grand mean FP/hr              (ALL folds)        : {grand_mean_fpr:.3f}")
if pass_mean_sens is not None:
    print(f"Grand mean event sensitivity (PASS folds only)  : {pass_mean_sens:.3f} ± {pass_std_sens:.3f}  <- use this one")
if pass_mean_fpr is not None:
    print(f"Grand mean FP/hr              (PASS folds only)  : {pass_mean_fpr:.3f}")
print(f"\nAli et al. (2024) true-LOPO comparator: 72-75% event sensitivity, "
      f"RF + 92 hand-crafted features, no hardware.")
if pass_mean_sens is not None:
    gap = 0.735 - pass_mean_sens   # midpoint of 72-75%, PASS-only basis
    print(f"Gap vs. Ali et al. midpoint (73.5%), PASS-only basis: {gap:+.3f} "
          f"({'below' if gap > 0 else 'above'} the comparator)")

summary = {
    'n_folds': len(rows),
    'collapse_pass_count': n_pass,
    'grand_mean_event_sensitivity_all':  round(grand_mean_sens, 4) if grand_mean_sens is not None else None,
    'grand_std_event_sensitivity_all':   round(grand_std_sens, 4)  if grand_std_sens  is not None else None,
    'grand_mean_fp_per_hour_all':        round(grand_mean_fpr, 4)  if grand_mean_fpr  is not None else None,
    'grand_mean_event_sensitivity_pass_only': round(pass_mean_sens, 4) if pass_mean_sens is not None else None,
    'grand_std_event_sensitivity_pass_only':  round(pass_std_sens, 4)  if pass_std_sens  is not None else None,
    'grand_mean_fp_per_hour_pass_only':       round(pass_mean_fpr, 4)  if pass_mean_fpr  is not None else None,
    'ali_et_al_comparator': {'low': 0.72, 'high': 0.75, 'method': 'true LOPO, RF, 92 hand-crafted features'},
    'per_fold': rows,
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {args.out}")
