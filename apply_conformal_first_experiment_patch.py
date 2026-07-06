#!/usr/bin/env python3
"""
apply_conformal_first_experiment_patch.py
============================================
Candidate E first experiment (Handoff_post_dann_scoping_to_implementation.md
sec4): split-conformal threshold selection for FPR control, C2-only (RC3).

Key design point, confirmed by reading build_dataset.py and eval_event_
level.py this session: X_val is ALREADY the calibration set this needs.
It's chronologically after train, before test, real/untouched, and never
used today to select the reported metric (only for early-stopping). No new
data-split or dataset rebuild required -- this is a pure eval-time patch to
eval_event_level.py, same pattern as the existing --smooth-windows /
--threshold-sweep additions (sec4d).

Applies one edit:

  1. src/evaluation/eval_event_level.py
       - --conformal / --conformal-alpha flags. Mutually exclusive with
         --threshold-sweep (sweep is exploratory; conformal is the
         principled replacement for reading a threshold off the sweep).
       - Inference/scoring logic (margin -> sigmoid -> optional smoothing)
         factored into _score_windows(X, label) so calibration and eval
         scores go through an IDENTICAL pipeline -- required for the
         exchangeability the guarantee depends on (sec4c).
       - When --conformal is set: loads X_val/y_val from the same
         per-patient _dataset_ann.npz already on disk, scores it, takes
         the non-seizure windows' scores as the nonconformity distribution,
         and picks the threshold via the standard finite-sample-corrected
         quantile: ceil((n+1)(1-alpha))/n -- NOT the naive (1-alpha)
         quantile -- which is what makes P(future non-seizure fires) <=
         alpha an EXACT guarantee under exchangeability, not approximate.
       - Overrides args.spike_threshold with the computed value; everything
         downstream (run_at_threshold, event-level metrics, collapse
         diagnostic, JSON output) is UNCHANGED -- it just receives a
         conformally-selected threshold instead of a manually-typed one.
       - Conformal metadata (alpha, n_calib_negative, quantile_level,
         selected_threshold) recorded in the output JSON for provenance.

Does NOT touch build_dataset.py / train_baseline.py -- E needs no new
training or dataset fields, only a different threshold-selection procedure
applied to checkpoints that already exist (chb10ft/13ft/15ft/16ft).

C2-ONLY, BY DESIGN (sec4c): exchangeability holds within one patient's own
val/test windows, not across patients. Do not run --conformal against a
pool/frozen-C1 checkpoint evaluated on a DIFFERENT --eval-patient -- that
is exactly the cross-patient case (RC1) this guarantee does not cover.

Run from ~/dissertation/ with akida_env activated:
    python3 apply_conformal_first_experiment_patch.py

Hard-refuses (exits nonzero, writes nothing) if any anchor text isn't found
exactly once -- re-run after checking whether the file changed since this
patch was written against the 2 July 2026 snapshot.
"""
import sys

PATH = 'src/evaluation/eval_event_level.py'


def patch_file(path, replacements):
    with open(path, 'r') as f:
        content = f.read()
    for i, (old, new) in enumerate(replacements):
        n = content.count(old)
        if n == 0:
            sys.exit(f"REFUSING: anchor #{i} not found in {path}.\n"
                      "File on disk doesn't match what this patch expects "
                      "-- no changes written to this file.")
        if n > 1:
            sys.exit(f"REFUSING: anchor #{i} matches {n} times in {path} "
                      "(expected exactly 1). No changes written.")
        content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Patched: {path}")


# ── 1. Flags ─────────────────────────────────────────────────────────────────
old_1 = '''parser.add_argument('--smooth-windows', type=int, default=0,
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
args = parser.parse_args()
assert os.path.exists(args.fbz), f"Model not found: {args.fbz}"'''

new_1 = '''parser.add_argument('--smooth-windows', type=int, default=0,
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
args = parser.parse_args()
assert os.path.exists(args.fbz), f"Model not found: {args.fbz}"
if args.conformal and args.threshold_sweep:
    parser.error('--conformal and --threshold-sweep are mutually exclusive '
                 '-- sweep is exploratory, conformal is the principled '
                 'selection procedure; run them separately.')'''

# ── 2. Factor scoring into a reusable function + calibration block ───────────
old_2 = """# ── Run inference ─────────────────────────────────────────────────────────────
print("\\nRunning SNN simulator inference...")
preds_raw = akida_model.predict(X_eval)
spike_counts = preds_raw.squeeze()  # (N,1,1,C) -> (N,C)
# new:
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
    print(f"\\n[Item 5] Applying causal {K}-window moving average to the "
          f"spike-margin ratio before thresholding (smoothing window "
          f"{K}s at this project's 1.0s window stride).")
    ratio = smoothed
def run_at_threshold(thr):"""

new_2 = '''# ── Run inference ─────────────────────────────────────────────────────────────
def _score_windows(X, label):
    """
    Shared scoring pipeline (margin -> sign-safe sigmoid -> optional causal
    smoothing), factored out so --conformal's calibration pass and the
    real eval pass go through IDENTICAL code -- a nonconformity score
    computed differently between the two would break the exchangeability
    the conformal guarantee depends on (sec4c). Behaviour for the eval
    pass is byte-for-byte unchanged from before this patch.
    """
    print(f"\\nRunning SNN simulator inference ({label})...")
    preds_raw = akida_model.predict(X)
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

# ── Candidate E: split-conformal threshold selection (sec4) ───────────────────
conformal_info = None
if args.conformal:
    assert 'X_val' in data.files, (
        f"{data_path} has no X_val split -- cannot calibrate. Should not "
        "happen for a standard build_dataset.py output; check the dataset."
    )
    if args.longctx:
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
    print(f"\\n[Conformal, Candidate E] Calibration set: {len(y_calib)} "
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


def run_at_threshold(thr):'''

# ── 3. Record conformal metadata in the output JSON ───────────────────────────
old_3 = """    json.dump({
        'fbz_path': args.fbz,
        'eval_patient': args.eval_patient,
        'spike_threshold': args.spike_threshold,
        'scaler_source': args.scaler_source if args.scaler_source else own_scaler_path,"""

new_3 = """    json.dump({
        'fbz_path': args.fbz,
        'eval_patient': args.eval_patient,
        'spike_threshold': args.spike_threshold,
        'conformal': conformal_info,
        'scaler_source': args.scaler_source if args.scaler_source else own_scaler_path,"""


if __name__ == '__main__':
    patch_file(PATH, [
        (old_1, new_1),
        (old_2, new_2),
        (old_3, new_3),
    ])
    print("\nAll patches applied successfully.")
    print("\nSanity check:")
    print("  python3 -m py_compile src/evaluation/eval_event_level.py")
    print("\nFirst screen (per sec4, C2-only -- point --fbz at an already-"
          "converted PERSONALISED checkpoint, e.g. chb10ft/13ft/15ft/16ft):")
    print("  python3 src/evaluation/eval_event_level.py --fbz "
          "results/seizure_model_chb10ft_v2_w4a4.fbz --eval-patient chb10 "
          "--conformal --conformal-alpha 0.05")
    print("  # repeat --conformal-alpha 0.01, and for chb13ft/chb15ft/chb16ft")
