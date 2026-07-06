"""
aggregate_g_multiseed.py
==========================
Candidate G phase 1, multi-seed collapse-stability check (post-sec8 six-
patient screen). Mirrors aggregate_breadth_eval.py's pattern exactly --
same file-naming convention, same mean/std-across-seeds computation, same
principle of reporting the Gate 0c collapse flag SEPARATELY from event
sensitivity rather than folding them into one verdict.

Motivation: the single-seed (123) six-patient screen found three patients
that collapse (chb13, chb15, chb20) in two DIFFERENT directions (chb13/
chb20 over-fire into a single ~100%-of-recording block; chb15 goes almost
completely non-responsive) -- a different, more concerning pattern than
DANN/CORAL/SSL's single consistent over-firing failure mode. Before
writing this up as a candidate verdict either way, this checks whether
that collapse behaviour is a stable property of G's feature set on these
three patients, or seed-sensitive training noise.

Usage (from ~/dissertation, after running seeds 123/42/2024 -- see the
rename-before-eval workflow that produces the _seed{N}_ naming this script
expects):
    python3 aggregate_g_multiseed.py
"""
import json
import glob
import re
import statistics as stats

PATTERN = 'results/event_results_seizure_model_multi_g_v2_w4a4_seed*_on_*.json'

FNAME_RE = re.compile(
    r'event_results_seizure_model_multi_g_v2_w4a4_seed(\d+)_on_(\w+)\.json'
)

# The single-seed (123) screen's collapse direction per patient, so the
# per-seed table below can be read against what's already known rather
# than re-derived from scratch each time.
KNOWN_DIRECTION = {
    'chb13': 'over-firing (100% block, seed123)',
    'chb15': 'under-firing / non-responsive (seed123)',
    'chb20': 'over-firing (100% block, seed123)',
}


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
        print(f"No files matched {PATTERN} -- check you're running from "
              "~/dissertation and that the rename-before-eval workflow "
              "was followed (seed must be in the .fbz filename before "
              "eval_event_level.py runs, not added after).")
        raise SystemExit(1)

    print(f"{'Patient':<8} {'n':>2}  {'EvtSens':>16}  {'FP/hr':>16}  "
          f"{'WinSpec':>16}  {'PosWinFrac':>14}  {'CollapseFlag':>14}")
    print("-" * 116)

    for patient in sorted(by_patient.keys()):
        runs = sorted(by_patient[patient], key=lambda r: int(r[0]))
        sens = [r['event_level']['event_sensitivity'] for _, r in runs
                if r.get('event_level', {}).get('event_sensitivity') is not None]
        fphr = [r['event_level']['fp_per_hour'] for _, r in runs
                if r.get('event_level', {}).get('fp_per_hour') is not None]
        winspec = [r['window_level']['specificity'] for _, r in runs
                   if r.get('window_level', {}).get('specificity') is not None]
        posfrac = [r['collapse_diagnostic']['positive_window_fraction'] for _, r in runs
                   if r.get('collapse_diagnostic', {}).get('positive_window_fraction') is not None]
        collapse_flags = [r.get('collapse_diagnostic', {}).get('pass') for _, r in runs]
        n_pass = sum(1 for c in collapse_flags if c)
        collapse_str = f"{n_pass}/{len(collapse_flags)} PASS"
        seeds_str = ','.join(s for s, _ in runs)

        print(f"{patient:<8} {len(runs):>2}  {fmt(sens):>16}  {fmt(fphr):>16}  "
              f"{fmt(winspec):>16}  {fmt(posfrac):>14}  {collapse_str:>14}   "
              f"(seeds: {seeds_str})")

        if patient in KNOWN_DIRECTION and len(runs) > 1:
            all_pass = n_pass == len(collapse_flags)
            all_fail = n_pass == 0
            if all_fail:
                verdict = ("STABLE collapse -- same failure across all seeds tested. "
                           f"Was: {KNOWN_DIRECTION[patient]}")
            elif all_pass:
                verdict = ("NOT REPRODUCED -- seed123's collapse did not recur at "
                           "the other seed(s). Was seed-sensitive training noise, "
                           "not a stable property of the G feature set on this "
                           "patient. Check specificity/FP-hr trend before reading "
                           "this as \"fixed\", though.")
            else:
                verdict = ("MIXED across seeds -- collapses on some seeds, not "
                           "others. Neither a clean confirm nor a clean disconfirm; "
                           "report as unstable/seed-sensitive, not as resolved "
                           "either direction.")
            print(f"         -> {verdict}")

    print("-" * 116)
    print("\nNOTE (same principle as aggregate_breadth_eval.py): CollapseFlag is "
          "Gate 0c's degeneracy check, independent of event sensitivity -- a high "
          "sensitivity with collapse FAIL still needs investigating on its own "
          "terms (this is exactly what happened with chb13/chb20's seed123 "
          "results: 0.667/1.000 sensitivity that turned out to be a single "
          "~100%-of-recording block, not genuine detection).")
