"""
aggregate_auprc_diagnostic.py
================================
Reads the event_results_*.json files produced by running
eval_event_level.py's window-level AUPRC diagnostic against C1 and C2
checkpoints for chb13/chb15/chb16, and prints a summary table with an
automatic interpretation of each result.

Interpretation logic, CORRECTED (previous version used a fixed 0.05 AUPRC
cutoff across all patients -- wrong, because AUPRC's meaningful "no-skill"
baseline is each patient's own seizure-window prevalence, and prevalence
varies substantially here: chb13 1.65%, chb15 2.74%, chb16 0.41% in the
run this was built against. A raw AUPRC of e.g. 0.16 is a MUCH stronger
result for chb16 (0.41% base rate, ~39x) than for chb15 (2.74% base rate,
~6x) despite chb15's raw number being higher -- comparing raw AUPRC across
patients with different prevalence is not a fair comparison).

Fixed logic: compute AUPRC / seizure_base_rate for each row (a "skill
ratio" -- how many times better than a no-skill classifier at this
patient's own prevalence) and interpret THAT, not the raw AUPRC, against
a fixed threshold. This requires reading each patient's true seizure-
window count / total window count from the JSON (n_events-derived fields
aren't quite it -- uses window-level counts, read from the dataset
directly via the eval JSON's window_level block if available, otherwise
requires --seizure-rates to be supplied manually since older JSON schemas
don't record window-level seizure count).

Usage:
    python3 aggregate_auprc_diagnostic.py --results-dir results
    python3 aggregate_auprc_diagnostic.py --results-dir results \\
        --seizure-rate chb13=0.0165 --seizure-rate chb15=0.0274 --seizure-rate chb16=0.0041
"""
import argparse, glob, json, os

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', default='results')
parser.add_argument('--seizure-rate', action='append', default=[],
                     help="Override/supply a patient's true seizure-window "
                          "base rate as patient=rate (e.g. chb13=0.0165). "
                          "Can be repeated. Falls back to the rates "
                          "confirmed this session if not supplied.")
args = parser.parse_args()

# Confirmed this session from actual eval_event_level.py output
# ("Eval set: N windows, M seizure windows") -- used as the default
# base rate for the skill-ratio calculation below. Override with
# --seizure-rate if these datasets are regenerated and the counts change.
DEFAULT_SEIZURE_RATES = {
    'chb13': 294 / 17816,
    'chb15': 592 / 21600,
    'chb16': 42 / 10258,
}
seizure_rates = dict(DEFAULT_SEIZURE_RATES)
for spec in args.seizure_rate:
    p, v = spec.split('=')
    seizure_rates[p] = float(v)

# (label, filename) pairs -- both patterns CONFIRMED real filenames
# (checked against actual `ls results/*.fbz` output this session):
# C1 = results/seizure_model_multi_v2_w4a4.fbz (frozen chb01/02/05-pool
# checkpoint). C2 = results/event_results_seizure_model_{p}ft_s256_v2_w4a4_on_{p}.json.
TARGETS = []
for p in ['chb13', 'chb15', 'chb16']:
    TARGETS.append(('C1 (frozen pool)', p, f'event_results_seizure_model_multi_v2_w4a4_on_{p}.json'))
    TARGETS.append(('C2 (per-patient ft)', p, f'event_results_seizure_model_{p}ft_s256_v2_w4a4_on_{p}.json'))

# Skill-ratio bands -- deliberately coarse, not precision thresholds.
# <1.5x: essentially chance-level ranking, regardless of raw AUPRC.
# 1.5-4x: weak but non-trivial signal.
# 4-10x: decent, real discrimination.
# >10x: strong.
def band(ratio):
    if ratio < 1.5:
        return "~chance-level ranking (retrain territory, regardless of raw AUPRC)"
    elif ratio < 4:
        return "weak but non-trivial ranking"
    elif ratio < 10:
        return "decent, real discrimination"
    else:
        return "strong discrimination"

rows = []
for cond, patient, fname in TARGETS:
    path = os.path.join(args.results_dir, fname)
    if not os.path.exists(path):
        rows.append((cond, patient, None, None, None, None, f"MISSING: {fname} not found -- run eval_event_level.py first"))
        continue
    with open(path) as f:
        r = json.load(f)
    auprc = r.get('window_level', {}).get('auprc')
    collapse_pass = r.get('collapse_diagnostic', {}).get('pass')
    base_rate = seizure_rates.get(patient)

    if auprc is None:
        rows.append((cond, patient, None, collapse_pass, base_rate, None, "AUPRC field missing -- was this run with the patched eval_event_level.py?"))
        continue
    if base_rate is None:
        rows.append((cond, patient, auprc, collapse_pass, None, None, f"No seizure base rate known for {patient} -- supply --seizure-rate {patient}=X"))
        continue

    ratio = auprc / base_rate
    ranking_read = band(ratio)
    if not collapse_pass and ratio >= 4:
        interp = f"{ranking_read}; collapse FAIL -> threshold misplaced (re-threshold territory)"
    elif not collapse_pass and ratio < 4:
        interp = f"{ranking_read}; collapse FAIL -> both ranking AND threshold are problems"
    elif collapse_pass and ratio < 1.5:
        interp = f"{ranking_read} despite collapse PASS -- PASS here likely reflects near-total silence, not a good operating point"
    else:
        interp = f"{ranking_read}; collapse PASS -> current operating point usable"
    rows.append((cond, patient, auprc, collapse_pass, base_rate, ratio, interp))

print(f"{'Condition':<20} {'Patient':<8} {'AUPRC':<7} {'BaseRate':<9} {'Ratio':<7} {'Collapse':<9} Interpretation")
print("-" * 130)
for cond, patient, auprc, collapse_pass, base_rate, ratio, interp in rows:
    auprc_s = f"{auprc:.4f}" if auprc is not None else "  --  "
    base_s = f"{base_rate:.4f}" if base_rate is not None else "  --   "
    ratio_s = f"{ratio:.1f}x" if ratio is not None else "  --  "
    collapse_s = ("PASS" if collapse_pass else "FAIL") if collapse_pass is not None else " -- "
    print(f"{cond:<20} {patient:<8} {auprc_s:<7} {base_s:<9} {ratio_s:<7} {collapse_s:<9} {interp}")

print("\nNote: 'Ratio' = AUPRC / this patient's own seizure-window base rate")
print("(i.e. how many times better than a no-skill classifier at this")
print("patient's own prevalence) -- NOT raw AUPRC. Raw AUPRC is not")
print("comparable across patients with different seizure-window prevalence.")
