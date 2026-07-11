"""
src/evaluation/eval_event_level.py
====================================
Software-simulator event-level evaluation for a converted SNN (.fbz) model.

Runs on WSL2 — NO physical AKD1000 required. Uses akida.Model(fbz_path)
WITHOUT .map(device), which runs the SNN simulator on CPU (same pattern
as convert_to_snn.py's own evaluation step).

Computes BOTH window-level and event-level (sliding_vote) metrics, since
window-level sensitivity for this project has consistently underreported
true clinical performance (chb03 window=0.377 / event=1.000 on the ANN).

Usage:
    python3 src/evaluation/eval_event_level.py \
        --fbz results/seizure_model_multi_noz_v2_w4a4.fbz \
        --eval-patient chb03

    python3 src/evaluation/eval_event_level.py \
        --fbz results/seizure_model_multi_noz_v2_w4a4.fbz \
        --eval-patient chb10

    # Sweep spike-count thresholds before settling on event-level default 0.5
    python3 src/evaluation/eval_event_level.py \
        --fbz results/seizure_model_multi_noz_v2_w4a4.fbz \
        --eval-patient chb03 --threshold-sweep
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import argparse, json, os, sys
import numpy as np
import akida

sys.path.insert(0, '.')
from src.evaluation.sliding_vote import event_level_metrics, collapse_diagnostic
from src.manifest import load_manifest, require_scaler_match
from sklearn.metrics import average_precision_score
parser = argparse.ArgumentParser()
parser.add_argument('--fbz',           required=True,
                    help="Path to converted .fbz SNN model")
parser.add_argument('--eval-patient',  required=True,
                    help="Patient tag whose X_test/y_test to evaluate on")
parser.add_argument('--spike-threshold', type=float, default=0.5,
                    help="Threshold on spike_count[:,1]/total ratio for "
                         "binary classification (default 0.5)")
parser.add_argument('--threshold-sweep', action='store_true',
                    help="Sweep thresholds 0.2-0.8 and report event-level "
                         "sensitivity/FPR at each, instead of a single run")
parser.add_argument('--gap-tolerance',   type=float, default=60.0)
parser.add_argument('--detection-fraction', type=float, default=0.3)
parser.add_argument('--min-sustained',   type=int, default=3)
parser.add_argument('--scaler-source', default=None,
                    help="Path to a scaler JSON to re-scale --eval-patient's "
                         "X_test under, OVERRIDING the scaler baked into its "
                         "own _dataset_ann.npz at build time. Use this when "
                         "evaluating a pool-trained model on a patient that "
                         "also contributed to the pool under a different "
                         "scaler (the chb06 dual-scaler bug, §5). Accepts "
                         "either a flat {scale,shift} JSON (e.g. chbXX_scaler.json) "
                         "or a pool scaler JSON with a 'per_patient' map "
                         "(e.g. multi_scaler.json) — the eval patient's own "
                         "entry is looked up automatically in the latter case.")
parser.add_argument('--lopo-full',      action='store_true',
                    help="Supervisor-directed LOPO session, 9 July 2026: "
                         "evaluate on the held-out patient's FULL recording "
                         "(data/processed/<patient>_dataset_lopo_full.npz, "
                         "from build_lopo_eval_set.py) instead of the "
                         "chronological 15% test slice. Matches standard "
                         "LOPO methodology and the Ali et al. (2024) "
                         "comparator directly. Mutually exclusive with "
                         "--longctx/--g-features.")
parser.add_argument('--longctx',        action='store_true')
parser.add_argument('--eval-batch-size', type=int, default=8000,
                    help="Fix, 10 July 2026 LOPO session: windows per "
                         "akida_model.predict() call. The simulator's peak "
                         "memory scales with batch size, so an unbatched "
                         "call on a full-recording eval set (200k+ windows) "
                         "OOM-killed the process for several patients. "
                         "Default 8000 keeps peak memory well bounded "
                         "regardless of eval-set size; lower it further if "
                         "a future patient still OOMs.")
parser.add_argument('--window-samples', type=int, default=512, choices=[512, 768])
parser.add_argument('--g-features',     action='store_true',
                    help='Evaluate a Candidate G checkpoint (relative-band-'
                         'power 3-channel dataset, Handoff_a_e_c_closed_to_'
                         'next_steps.md sec8 phase 1). Mutually exclusive '
                         'with --longctx.')
parser.add_argument('--smooth-windows', type=int, default=0,
                    help="Item 5 (score-level temporal smoothing): apply a "
                         "CAUSAL moving average over this many consecutive "
                         "windows to the continuous spike-margin ratio "
                         "BEFORE thresholding. Default 0 = off (unchanged "
                         "behaviour). Causal only (uses window i and the "
                         "(K-1) windows before it, never future windows) -- "
                         "a centered average would leak future information "
                         "into a real-time decision and unrealistically "
                         "flatter detection latency. Distinct from the "
                         "rejected longctx channel design (Gate 2): this "
                         "adds zero trainable parameters and requires no "
                         "fine-tuning data -- it's a pure inference-time "
                         "signal-processing step on the existing model's "
                         "own output, same spirit as sliding_vote.py's "
                         "gap-merging, just operating on the continuous "
                         "score instead of the binarised decision.")
parser.add_argument('--conformal', action='store_true',
                    help="Candidate E (Handoff_post_dann_scoping_to_"
                         "implementation.md sec4): select the operating "
                         "threshold via split-conformal FPR-control "
                         "calibration instead of --spike-threshold. "
                         "Calibrates on this patient's OWN X_val split "
                         "(chronologically before X_test, never used to "
                         "pick the reported metric). C2-ONLY: exchangeability "
                         "holds within one patient's own val/test windows, "
                         "NOT across patients -- do not use against a "
                         "pool/frozen-C1 checkpoint evaluated on a "
                         "different --eval-patient. Mutually exclusive "
                         "with --threshold-sweep.")
parser.add_argument('--conformal-alpha', type=float, default=0.05,
                    help="Target miscoverage for --conformal: guarantees "
                         "P(future non-seizure window fires) <= alpha "
                         "(default 0.05). First screen also tries 0.01 "
                         "for a tighter guarantee.")
parser.add_argument('--force-scaler-mismatch', action='store_true',
                    help="C2 compounding item 2 (amplitude-calibration + "
                         "C2 stacking): Gate 1b hard-refuses any "
                         "--scaler-source override whose scale/shift "
                         "doesn't match the checkpoint's own recorded "
                         "training scaler -- correct behaviour for "
                         "catching an ACCIDENTAL mismatch (the chb06 bug "
                         "class), but it also blocks a DELIBERATE, "
                         "disclosed scale shift for a controlled "
                         "experiment (e.g. stacking sec3b's amplitude-"
                         "calibration scaler on top of an already fine-"
                         "tuned checkpoint). This flag converts that "
                         "hard-exit into a loud, logged override -- it "
                         "does not silence or remove the check, it "
                         "disclosed-bypasses it for this run only. "
                         "Requires --scaler-source; refused otherwise.")
args = parser.parse_args()
assert os.path.exists(args.fbz), f"Model not found: {args.fbz}"
if args.conformal and args.threshold_sweep:
    parser.error('--conformal and --threshold-sweep are mutually exclusive '
                 '-- sweep is exploratory, conformal is the principled '
                 'selection procedure; run them separately.')
if args.force_scaler_mismatch and not args.scaler_source:
    parser.error('--force-scaler-mismatch requires --scaler-source -- '
                 'there is no mismatch to force past without an override '
                 'scaler in the first place.')

if sum([args.longctx, args.g_features, args.lopo_full]) > 1:
    sys.exit("ERROR: --longctx, --g-features, and --lopo-full are mutually "
             "exclusive.")
if args.longctx:
    data_path = (f'data/processed/{args.eval_patient}_dataset_longctx_w'
                 f'{args.window_samples}.npz')
elif args.g_features:
    data_path = f'data/processed/{args.eval_patient}_dataset_g.npz'
elif args.lopo_full:
    data_path = f'data/processed/{args.eval_patient}_dataset_lopo_full.npz'
else:
    data_path = f'data/processed/{args.eval_patient}_dataset_ann.npz'
assert os.path.exists(data_path), (
    f"Dataset not found: {data_path}"
    + ("\nRun: python3 src/preprocessing/build_lopo_eval_set.py "
       f"--patient {args.eval_patient}" if args.lopo_full else ""))

print(f"Model        : {args.fbz}")
print(f"Eval patient : {args.eval_patient}")

# ── Load model (simulator mode — no .map(device), runs on CPU) ──────────────
akida_model = akida.Model(args.fbz)

# ── Load eval data ────────────────────────────────────────────────────────────
data = np.load(data_path)
assert 'X_test' in data.files, f"{data_path} has no X_test split"
if args.longctx or args.g_features:
    X_eval = data['X_test'].astype('float32', copy=False)
else:
    X_eval = data['X_test'][..., np.newaxis].astype('float32', copy=False)
y_eval = data['y_test']
print(f"Eval set     : {len(y_eval)} windows, {int(y_eval.sum())} seizure windows")

# ── Gate 0b: optional scaler override (chb06 dual-scaler fix, §5) ───────────
if args.longctx:
    own_scaler_path = (f'data/processed/{args.eval_patient}_scaler_longctx_w'
                       f'{args.window_samples}.json')
elif args.g_features:
    own_scaler_path = f'data/processed/{args.eval_patient}_scaler_g.json'
else:
    own_scaler_path = f'data/processed/{args.eval_patient}_scaler.json'
assert os.path.exists(own_scaler_path), f"Missing {own_scaler_path}"
with open(own_scaler_path) as f:
    own_scaler = json.load(f)

if args.scaler_source and args.g_features:
    sys.exit("ERROR: --scaler-source is not supported with --g-features -- "
             "G's scaler is per-channel format ({ch0_min,...}), not the flat "
             "{scale,shift} format --scaler-source's override logic assumes "
             "(same limitation --longctx already has, pre-existing). Not "
             "needed for phase 1 (frozen-pool eval on held-out patients, no "
             "dual-scaler correction involved).")
if args.scaler_source:
    assert os.path.exists(args.scaler_source), f"Scaler not found: {args.scaler_source}"
    with open(args.scaler_source) as f:
        src = json.load(f)
    if 'per_patient' in src:
        if args.eval_patient not in src['per_patient']:
            sys.exit(f"ERROR: {args.eval_patient} has no entry in "
                      f"{args.scaler_source}'s per_patient map. "
                      f"Available: {list(src['per_patient'])}")
        override_scaler = src['per_patient'][args.eval_patient]
    else:
        override_scaler = src

    print(f"\n[Gate 0b scaler override]")
    print(f"  Eval data was built under : {own_scaler_path} "
          f"(scale={own_scaler['scale']:.2f}, shift={own_scaler['shift']:.2f})")
    print(f"  Re-scaling under          : {args.scaler_source} "
          f"(scale={override_scaler['scale']:.2f}, shift={override_scaler['shift']:.2f})")

    # Undo the npz's baked-in scaling (back to raw EEG units), then reapply
    # the override scaler — the exact correction §5 confirmed by hand.
    X_raw  = (X_eval - own_scaler['shift']) / own_scaler['scale']
    X_eval = np.clip(X_raw * override_scaler['scale'] + override_scaler['shift'],
                      0.0, 255.0).astype('float32')
else:
    print(f"\n[Scaler] using {own_scaler_path} as baked into the npz "
          f"(no --scaler-source override given — unchanged default behaviour)")
# ── Gate 1b: refuse to run if the model's training scaler doesn't match
# the scaler actually being applied to this eval run ────────────────────
model_manifest = load_manifest(args.fbz, required=False)
if model_manifest is None:
    print(f"\n[Gate 1b] WARNING: no manifest for {args.fbz} — scaler "
          "consistency cannot be verified. Proceeding on the strength of "
          "--scaler-source alone. Regenerate this .fbz with the current "
          "convert_to_snn.py to get this check.")
else:
    model_scaler = model_manifest.get('scaler')
    if model_scaler is None:
        print(f"\n[Gate 1b] WARNING: {args.fbz}'s manifest has no scaler "
              "recorded — cannot verify.")
    else:
        if 'per_patient' in model_scaler:
            if args.eval_patient not in model_scaler['per_patient']:
                print(f"\n[Gate 1b] NOTE: {args.eval_patient} was never one of "
                      f"{args.fbz}'s pool constituents — no pool-relative "
                      "scaler to check against. This is the normal "
                      "cross-patient generalisation case (e.g. chb10), not "
                      "the chb06 dual-scaler bug (which only applies to "
                      "patients that played BOTH roles). Proceeding without "
                      "a scaler check.")
                expected_scaler = None
            else:
                expected_scaler = model_scaler['per_patient'][args.eval_patient]
        else:
            expected_scaler = model_scaler

        if expected_scaler is not None:
            if args.longctx or args.g_features:
                _tag = 'Longctx' if args.longctx else 'Candidate G'
                print(f"\n[Gate 1b] {_tag} run — per-channel scaler; "
                      "require_scaler_match skipped (Gate 2c verified at training).")
            elif args.force_scaler_mismatch:
                actual_scaler = override_scaler if args.scaler_source else own_scaler
                print(f"\n[Gate 1b] *** DELIBERATE OVERRIDE — check bypassed "
                      f"by --force-scaler-mismatch ***")
                print(f"  Model's own recorded training scaler : "
                      f"scale={expected_scaler['scale']:.2f}, "
                      f"shift={expected_scaler['shift']:.2f}")
                print(f"  Scaler actually applied this run      : "
                      f"scale={actual_scaler['scale']:.2f}, "
                      f"shift={actual_scaler['shift']:.2f}")
                print(f"  This is NOT the chb06 dual-scaler bug being "
                      "re-triggered -- it's a disclosed, controlled scale "
                      "shift for a stacking experiment. Treat any result "
                      "from this run as exploratory: the model has never "
                      "seen this input distribution during training.")
            else:
                actual_scaler = override_scaler if args.scaler_source else own_scaler
                require_scaler_match(
                    expected_scaler, actual_scaler,
                    context=f"{args.fbz} (trained) vs. eval input for "
                             f"{args.eval_patient} (this run)",
                )
                print(f"\n[Gate 1b] Scaler consistency verified against "
                      f"{args.fbz}'s manifest ✓")
# ── Run inference ─────────────────────────────────────────────────────────────
def _score_windows(X, label):
    """
    Shared scoring pipeline (margin -> sign-safe sigmoid -> optional causal
    smoothing), factored out so --conformal's calibration pass and the
    real eval pass go through IDENTICAL code -- a nonconformity score
    computed differently between the two would break the exchangeability
    the conformal guarantee depends on (sec4c). Behaviour for the eval
    pass is byte-for-byte unchanged from before this patch.
    """
    print(f"\nRunning SNN simulator inference ({label})...")
    # Batched (fix, 10 July 2026 LOPO session) -- see module docstring of
    # apply_lopo_eval_batching_patch.py for why this was necessary.
    _batch = args.eval_batch_size
    _preds_parts = []
    for _start in range(0, len(X), _batch):
        _preds_parts.append(akida_model.predict(X[_start:_start + _batch]))
    preds_raw = np.concatenate(_preds_parts, axis=0)
    del _preds_parts
    spike_counts = preds_raw.squeeze()  # (N,1,1,C) -> (N,C)
    # spike_counts are signed potentials (unconstrained final Dense readout),
    # not guaranteed non-negative spike counts. The old ratio=count1/total
    # formula silently INVERTS the decision whenever total<0 (dividing an
    # inequality by a negative number flips it) -- confirmed present at 0.14%
    # of windows on chb10ft, 23.7% on the Condition 4 checkpoint. Fixed via a
    # sigmoid of the margin: sign-safe, bounded (0,1), monotonic, and equal to
    # the natural decision boundary (predict positive iff count1>count0) at
    # the existing default threshold 0.5 -- no CLI interface change needed.
    margin = spike_counts[:, 1] - spike_counts[:, 0]
    ratio = 1.0 / (1.0 + np.exp(-margin))
    # Item 5: optional causal moving-average smoothing on the continuous score,
    # applied BEFORE thresholding and BEFORE any of the existing event-level /
    # collapse-diagnostic logic -- those stay completely unchanged downstream,
    # they just now see a smoothed ratio array if --smooth-windows > 0.
    if args.smooth_windows > 1:
        K = args.smooth_windows
        csum = np.cumsum(np.insert(ratio, 0, 0.0))
        smoothed = np.empty_like(ratio)
        for i in range(len(ratio)):
            lo = max(0, i - K + 1)
            smoothed[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
        print(f"[Item 5] Applying causal {K}-window moving average to the "
              f"spike-margin ratio before thresholding ({label}).")
        ratio = smoothed
    return ratio


ratio = _score_windows(X_eval, label='eval/test set')

# ── Action item 4: window-level AUPRC diagnostic (threshold-independent) ────
# Pure re-scoring of the existing continuous score against ground truth --
# no new inference. Computed once, before any threshold is chosen (spike-
# threshold, sweep, or conformal), so it's identical across all three modes
# for a given checkpoint/patient pair. Purpose: distinguish "genuinely
# inert" (poor ranking -- low AUPRC) from "threshold misplaced" (ranking is
# fine, the chosen cut point isn't) for chb13/chb15/chb16 specifically
# (Handoff_items1to3_done_item4_next.md sec4).
window_auprc = float(average_precision_score(y_eval, ratio))
print(f"\n[Item 4] Window-level AUPRC (threshold-independent): {window_auprc:.4f}")

# ── Candidate E: split-conformal threshold selection (sec4) ───────────────────
conformal_info = None
if args.conformal:
    assert 'X_val' in data.files, (
        f"{data_path} has no X_val split -- cannot calibrate. Should not "
        "happen for a standard build_dataset.py output; check the dataset."
    )
    if args.longctx or args.g_features:
        X_calib = data['X_val'].astype('float32')
    else:
        X_calib = data['X_val'][..., np.newaxis].astype('float32')
    y_calib = data['y_val']

    # Same scaler treatment X_eval got, so calibration and eval scores come
    # from an identical pipeline (required for exchangeability, sec4c).
    if args.scaler_source:
        X_calib_raw = (X_calib - own_scaler['shift']) / own_scaler['scale']
        X_calib = np.clip(
            X_calib_raw * override_scaler['scale'] + override_scaler['shift'],
            0.0, 255.0).astype('float32')

    n_calib_neg = int((y_calib == 0).sum())
    if n_calib_neg == 0:
        sys.exit(
            f"ERROR: {args.eval_patient}'s X_val has zero non-seizure "
            "windows -- cannot calibrate an FPR-control threshold with no "
            "negative-class calibration data."
        )
    print(f"\n[Conformal, Candidate E] Calibration set: {len(y_calib)} "
          f"windows from {args.eval_patient}'s OWN X_val split "
          f"({n_calib_neg} non-seizure, {int((y_calib == 1).sum())} seizure "
          "-- only the non-seizure windows feed FPR-control calibration).")

    calib_ratio = _score_windows(X_calib, label='calibration set')
    neg_scores  = calib_ratio[y_calib == 0]

    # Finite-sample-corrected quantile: ceil((n+1)(1-alpha))/n, NOT the naive
    # (1-alpha) quantile -- this correction is what makes the guarantee
    # EXACT under exchangeability, not just approximate.
    n = len(neg_scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - args.conformal_alpha)) / n)
    conformal_threshold = float(np.quantile(neg_scores, q_level, method='higher'))

    print(f"[Conformal] alpha={args.conformal_alpha}  n_calib_neg={n}  "
          f"quantile_level={q_level:.4f}  threshold={conformal_threshold:.6f}")
    print(f"[Conformal] Guarantee: P(future non-seizure window from "
          f"{args.eval_patient} fires) <= {args.conformal_alpha}, under "
          f"exchangeability of {args.eval_patient}'s own val/test windows. "
          "Does NOT hold across patients (sec4c) -- do not reuse this "
          "threshold on a different --eval-patient.")

    args.spike_threshold = conformal_threshold
    conformal_info = {
        'alpha': args.conformal_alpha,
        'n_calib_negative': n,
        'quantile_level': round(q_level, 6),
        'selected_threshold': round(conformal_threshold, 6),
    }


def run_at_threshold(thr):
    y_pred = (ratio >= thr).astype(np.int32)

    tp = int(((y_pred == 1) & (y_eval == 1)).sum())
    fp = int(((y_pred == 1) & (y_eval == 0)).sum())
    fn = int(((y_pred == 0) & (y_eval == 1)).sum())
    tn = int(((y_pred == 0) & (y_eval == 0)).sum())
    win_sens = tp / (tp + fn) if (tp + fn) > 0 else None
    win_spec = tn / (tn + fp) if (tn + fp) > 0 else None
    n_neg = tn + fp
    win_fpr_hr = fp / (n_neg * 2.0 / 3600) if n_neg > 0 else None

    window_s_actual = (args.window_samples / 256.0) if args.longctx else 2.0
    ev = event_level_metrics(
        y_eval, y_pred,
        window_s=window_s_actual, overlap=0.5,
        detection_fraction=args.detection_fraction,
        min_sustained_windows=args.min_sustained,
        gap_tolerance_s=args.gap_tolerance,
    )

    # Gate 0c — collapse diagnostic, now standard on every evaluation
    collapse = collapse_diagnostic(
        y_pred, window_specificity=win_spec, y_true=y_eval, step_s=1.0,
        gap_tolerance_s=args.gap_tolerance,
    )

    return win_sens, win_spec, win_fpr_hr, ev, collapse

if args.threshold_sweep:
    print(f"\n=== Threshold sweep ({args.eval_patient}) ===")
    print(f"  {'Thr':>7}  {'WinSens':>8}  {'WinSpec':>8}  {'WinFPRhr':>9}  "
          f"{'EvtSens':>8}  {'EvtDet':>7}  {'FPevt/hr':>9}  {'MeanLat':>8}  {'Collapse':>8}")
    print("  " + "-" * 90)
    for thr in [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8,
                0.85, 0.9, 0.92, 0.95, 0.97, 0.99,
                0.992, 0.995, 0.997, 0.999, 0.9995, 0.9999]:
        ws, wsp, wfpr, ev, collapse = run_at_threshold(thr)
        es = ev['event_sensitivity']
        es_s = f"{es:.3f}" if es is not None else "  N/A"
        fpr_s = f"{ev['fp_per_hour']:.2f}" if ev['fp_per_hour'] is not None else "  N/A"
        lat = ev['mean_latency_s']
        lat_s = f"{lat:.2f}" if lat is not None else "  N/A"
        flag = "PASS" if collapse['pass'] else "FAIL"
        print(f"  {thr:>7.4f}  {ws:>8.4f}  {wsp:>8.4f}  {wfpr:>9.2f}  "
              f"{es_s:>8}  {ev['n_detected']}/{ev['n_events']:>5}  {fpr_s:>9}  {lat_s:>8}  {flag:>8}")
    sys.exit(0)

win_sens, win_spec, win_fpr_hr, ev, collapse = run_at_threshold(args.spike_threshold)

print(f"\n=== Window-level (threshold={args.spike_threshold}) ===")
print(f"  Sensitivity : {win_sens:.4f}")
print(f"  Specificity : {win_spec:.4f}")
print(f"  FPR / hour  : {win_fpr_hr:.2f}")

print(f"\n=== Event-level (gap={args.gap_tolerance}s, "
      f"frac={args.detection_fraction}, sustained={args.min_sustained}) ===")
print(f"  Event sensitivity : {ev['event_sensitivity']}")
print(f"  Events detected   : {ev['n_detected']}/{ev['n_events']}")
print(f"  False positive events : {ev['false_positives']}")
print(f"  FP events / hour  : {ev['fp_per_hour']}")
print(f"  Total hours       : {ev['total_hours']:.2f}")
print(f"  Detection latency (s) : {ev['latencies_s']}")
print(f"  Mean latency (s)      : {ev['mean_latency_s']}")
print(f"  Undetected events     : {ev['n_undetected']}")

print(f"\n=== Collapse diagnostic (Gate 0c, §4) ===")
print(f"  Positive-window fraction : {collapse['positive_window_fraction']}")
print(f"  Predicted blocks         : {collapse['n_predicted_blocks']}")
print(f"  Largest block fraction   : {collapse['largest_block_fraction']}")
flag = "PASS" if collapse['pass'] else "FAIL"
print(f"  Verdict                  : {flag}")
if not collapse['pass']:
    for r in collapse['reasons']:
        print(f"    - {r}")

out_path = (f"results/event_results_"
            f"{os.path.basename(args.fbz).replace('.fbz','')}"
            f"_on_{args.eval_patient}.json")
with open(out_path, 'w') as f:
    json.dump({
        'fbz_path': args.fbz,
        'eval_patient': args.eval_patient,
        'spike_threshold': args.spike_threshold,
        'conformal': conformal_info,
        'force_scaler_mismatch': bool(args.force_scaler_mismatch),
        'scaler_source': args.scaler_source if args.scaler_source else own_scaler_path,
        'collapse_diagnostic': collapse,
        'window_level': {
            'sensitivity': round(win_sens, 4) if win_sens is not None else None,
            'specificity': round(win_spec, 4) if win_spec is not None else None,
            'fpr_per_hour': round(win_fpr_hr, 2) if win_fpr_hr is not None else None,
            'auprc': round(window_auprc, 4),
        },
        'event_level': {
            'event_sensitivity': ev['event_sensitivity'],
            'n_events': ev['n_events'],
            'n_detected': ev['n_detected'],
            'false_positives': ev['false_positives'],
            'fp_per_hour': ev['fp_per_hour'],
            'total_hours': ev['total_hours'],
            'gap_tolerance_s': args.gap_tolerance,
            'detection_fraction': args.detection_fraction,
            'min_sustained_windows': args.min_sustained,
            'latencies_s': ev['latencies_s'],
            'mean_latency_s': ev['mean_latency_s'],
            'n_undetected': ev['n_undetected']
        },
    }, f, indent=2)
print(f"\nSaved: {out_path}")
