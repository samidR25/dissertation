"""
aggregate_g_phase1.py
======================
Candidate G phase 1 (Handoff_a_e_c_closed_to_next_steps.md sec8) results
aggregator. Reads the six per-patient event_results_*.json files
eval_event_level.py writes directly -- no terminal-log parsing -- and
formats them into the same table style A's and C's six-patient screens
used, so the result drops straight into the next handoff.

Single-seed by design (matches A/C's first-screen discipline: "screen
cheap before confirming" -- multi-seed confirmation is legitimate
follow-up work, not a phase-1 requirement).

Usage (from ~/dissertation, after running eval_event_level.py --g-features
for all six patients):
    python3 aggregate_g_phase1.py
"""
import json
import glob
import re

PATTERN = 'results/event_results_seizure_model_multi_g_v2_w4a4_on_*.json'
FNAME_RE = re.compile(r'_on_(\w+)\.json')

# C1 baseline event sensitivity, per patient -- from Handoff_a_e_c_closed_
# to_next_steps.md sec1 (A's screen table). Hardcoded here (not re-derived)
# because these are already-confirmed numbers from a prior session, not
# something this script should be silently re-measuring.
C1_BASELINE = {
    'chb03': (0.000, 'structural'),
    'chb10': (0.500, ''),
    'chb13': (0.833, 'FAIL, over-firing'),
    'chb15': (0.000, 'under-fire'),
    'chb16': (0.000, 'PASS, inert'),
    'chb20': (0.000, ''),
}

# chb13 result under the three already-closed learned-representation
# candidates (sec4's cross-candidate table) -- reprinted here so G's
# chb13 result can be read directly against the pattern it's testing,
# without flipping back to the handoff mid-analysis.
CHB13_PRIOR = {
    'DANN':        (0.167, 'over-firing', 'spec 0.4518-0.8548'),
    'CORAL':       (0.167, 'over-firing', 'spec 0.5259-0.6184'),
    'SSL-pretrain':(0.167, 'over-firing', 'spec 0.7752'),
}


def load_all():
    by_patient = {}
    for path in sorted(glob.glob(PATTERN)):
        m = FNAME_RE.search(path)
        if not m:
            print(f"  [skip] couldn't parse filename: {path}")
            continue
        patient = m.group(1)
        with open(path) as f:
            by_patient[patient] = json.load(f)
    return by_patient


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if v is not None else "  N/A"


if __name__ == '__main__':
    by_patient = load_all()
    if not by_patient:
        print(f"No files matched {PATTERN} -- check you're running from "
              "~/dissertation and that all six eval_event_level.py "
              "--g-features runs completed.")
        raise SystemExit(1)

    expected = set(C1_BASELINE)
    got = set(by_patient)
    missing = expected - got
    extra = got - expected
    if missing:
        print(f"WARNING: missing results for {sorted(missing)} -- table "
              "below is partial, do not treat the grand mean as final "
              "until all six are present.")
    if extra:
        print(f"NOTE: unexpected extra patient(s) in results dir: "
              f"{sorted(extra)} (included below, not part of the six-"
              "patient screen set)")

    print(f"\n{'Patient':<8} {'C1 baseline':<20} {'G phase 1':<45} {'Collapse':<10}")
    print("-" * 90)

    grand_c1, grand_g = [], []
    for patient in sorted(C1_BASELINE) + sorted(extra):
        c1_sens, c1_note = C1_BASELINE.get(patient, (None, ''))
        c1_str = f"{fmt(c1_sens)}" + (f" ({c1_note})" if c1_note else "")

        r = by_patient.get(patient)
        if r is None:
            print(f"{patient:<8} {c1_str:<20} {'-- not run --':<45}")
            continue

        ev = r.get('event_level', {})
        collapse = r.get('collapse_diagnostic', {})
        sens = ev.get('event_sensitivity')
        n_det = ev.get('n_detected')
        n_ev = ev.get('n_events')
        fphr = ev.get('fp_per_hour')
        winspec = r.get('window_level', {}).get('specificity')
        flag = 'PASS' if collapse.get('pass') else 'FAIL'

        g_str = (f"{fmt(sens)} ({n_det}/{n_ev}) {flag} -- "
                f"spec {fmt(winspec, 4)}, FP/hr {fmt(fphr, 2)}")

        print(f"{patient:<8} {c1_str:<20} {g_str:<45} {flag:<10}")

        if c1_sens is not None:
            grand_c1.append(c1_sens)
        if sens is not None:
            grand_g.append(sens)

    print("-" * 90)
    if grand_c1 and grand_g and len(grand_c1) == len(grand_g):
        print(f"Grand mean -- C1 baseline: {sum(grand_c1)/len(grand_c1):.3f}   "
              f"G phase 1: {sum(grand_g)/len(grand_g):.3f}")
    else:
        print("Grand mean not computed -- incomplete results (see warnings above).")

    # ── chb13 headline comparison (sec4/sec8's actual question) ────────────
    print(f"\n{'='*60}")
    print("chb13 -- the cross-candidate motivating case (sec4/sec8)")
    print(f"{'='*60}")
    for name, (s, direction, spec) in CHB13_PRIOR.items():
        print(f"  {name:<14}: {s:.3f}  {direction:<15}  ({spec})")
    if 'chb13' in by_patient:
        r = by_patient['chb13']
        ev = r.get('event_level', {})
        collapse = r.get('collapse_diagnostic', {})
        winspec = r.get('window_level', {}).get('specificity')
        sens = ev.get('event_sensitivity')
        flag = 'PASS' if collapse.get('pass') else 'FAIL'
        print(f"  {'G phase 1':<14}: {fmt(sens)}  "
              f"{'PASS' if (sens or 0) > 0.3 else '(same regime as prior three?)':<15}  "
              f"(spec {fmt(winspec, 4)}, collapse {flag})")
        print()
        if sens is not None and abs(sens - 0.167) < 1e-6:
            print("  NOTE: chb13 landed at the EXACT same event sensitivity (1/6) "
                  "as DANN/CORAL/SSL-pretrain. Per sec4's own logic, a fourth "
                  "mechanistically distinct intervention (this one non-learned, "
                  "no representation-learning at all) landing at the identical "
                  "number would be strong further evidence the chb13 problem "
                  "isn't representational -- worth flagging explicitly in the "
                  "next handoff rather than treated as 'another negative result'.")
        elif sens is not None and sens > 0.3:
            print("  NOTE: chb13 responded DIFFERENTLY under G than under DANN/"
                  "CORAL/SSL-pretrain -- per sec8, this is the 'genuinely new "
                  "finding' branch. Check window specificity and the collapse "
                  "diagnostic carefully before reading this as a genuine win "
                  "(same caution the project has applied to every other "
                  "candidate's chb13 result).")
    else:
        print("  G phase 1     : -- not run yet --")
