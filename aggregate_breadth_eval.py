"""
aggregate_breadth_eval.py
==========================
Reads the actual per-seed/per-patient event_results_*.json artifacts
written directly by eval_event_level.py -- no terminal-log parsing,
no intermediate summarization. Computes mean/std across the 5 pool
seeds for each held-out patient, and flags the Gate 0c collapse
diagnostic SEPARATELY from event sensitivity, since they answer
different questions and must never be conflated into one verdict.

Usage (from ~/dissertation):
    python3 aggregate_breadth_eval.py
"""
import json
import glob
import re
import statistics as stats

PATTERN = 'results/event_results_seizure_model_multi_v2_w4a4_seed*_on_*.json'

FNAME_RE = re.compile(
    r'event_results_seizure_model_multi_v2_w4a4_seed(\d+)_on_(\w+)\.json'
)


def load_all():
    by_patient = {}
    for path in sorted(glob.glob(PATTERN)):
        m = FNAME_RE.search(path)
        if not m:
            print(f"  [skip] couldn't parse filename: {path}")
            continue
        seed, patient = m.group(1), m.group(2)
        with open(path) as f:
            data = json.load(f)
        by_patient.setdefault(patient, []).append((seed, data))
    return by_patient


def fmt(vals):
    if not vals:
        return "  N/A"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{stats.mean(vals):.3f} ± {stats.pstdev(vals):.3f}"


if __name__ == '__main__':
    by_patient = load_all()
    if not by_patient:
        print(f"No files matched {PATTERN} -- check you're running from ~/dissertation")
        raise SystemExit(1)

    print(f"{'Patient':<8} {'n':>2}  {'EvtSens':>14}  {'FP/hr':>16}  "
          f"{'WinSpec':>14}  {'PosWinFrac':>14}  {'CollapseFlag':>14}")
    print("-" * 110)

    grand_sens, grand_fphr = [], []

    for patient in sorted(by_patient.keys()):
        runs = by_patient[patient]
        sens = [r['event_level']['event_sensitivity'] for _, r in runs
                if r.get('event_level', {}).get('event_sensitivity') is not None]
        fphr = [r['event_level']['fp_per_hour'] for _, r in runs
                if r.get('event_level', {}).get('fp_per_hour') is not None]
        lat = [r['event_level']['mean_latency_s'] for _, r in runs
               if r.get('event_level', {}).get('mean_latency_s') is not None]
        winspec = [r['window_level']['specificity'] for _, r in runs
                   if r.get('window_level', {}).get('specificity') is not None]
        posfrac = [r['collapse_diagnostic']['positive_window_fraction'] for _, r in runs
                   if r.get('collapse_diagnostic', {}).get('positive_window_fraction') is not None]
        collapse_flags = [r.get('collapse_diagnostic', {}).get('pass') for _, r in runs]
        n_pass = sum(1 for c in collapse_flags if c)
        collapse_str = f"{n_pass}/{len(collapse_flags)} PASS"

        print(f"{patient:<8} {len(runs):>2}  {fmt(sens):>14}  {fmt(fphr):>16}  "
              f"{fmt(winspec):>14}  {fmt(posfrac):>14}  {collapse_str:>14}")

        grand_sens.extend(sens)
        grand_fphr.extend(fphr)

    print("-" * 110)
    print(f"Grand mean across all patients/seeds -- "
          f"sensitivity: {fmt(grand_sens)}   FP/hr: {fmt(grand_fphr)}")
    print("\nNOTE: 'CollapseFlag' is Gate 0c's degeneracy check (model hasn't gone "
          "all-positive/all-negative). It is INDEPENDENT of whether the model "
          "actually detects seizures well -- a high event sensitivity with a "
          "collapse FAIL still needs investigating on its own terms, and a collapse "
          "PASS with zero detections is not a good result just because it passed.")
