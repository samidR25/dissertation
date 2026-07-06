#!/usr/bin/env python3
"""
apply_auprc_diagnostic_patch.py
============================================
C2 compounding, action item 4 (Handoff_items1to3_done_item4_next.md sec4):
window-level AUPRC diagnostic.

Purpose: distinguish "genuinely inert" (the model's continuous ranking of
seizure vs. non-seizure windows is itself poor -- low AUPRC, e.g. chb15's
suspected profile) from "threshold misplaced" (ranking is fine, the chosen
operating point isn't, e.g. chb16ft's earlier FPR/FP-hr column mix-up) for
chb13/chb15/chb16 specifically. Pure re-scoring against the continuous
spike-margin score (`ratio`) already computed by `_score_windows()` -- no
new inference, no new training, and threshold-independent by construction
(average_precision_score never sees `--spike-threshold`, `--conformal`, or
the sweep loop). Computed once, immediately after `ratio` is produced, so
it is identical across single-threshold, --threshold-sweep, and --conformal
runs for a given checkpoint/patient pair.

Applies three narrow, additive changes -- nothing existing is removed or
reordered:

  1. src/evaluation/eval_event_level.py
       - New import: sklearn.metrics.average_precision_score.
       - `window_auprc = average_precision_score(y_eval, ratio)` computed
         right after `ratio = _score_windows(...)`, before the conformal
         block, the threshold-sweep block, or the single-threshold run --
         so it is ALWAYS computed, not flag-gated, and identical regardless
         of which of those three modes is used downstream.
       - Printed immediately (single-threshold, sweep, and conformal modes
         all pass through this print, since it sits upstream of all three).
       - Added to the output JSON's 'window_level' block as 'auprc' --
         JSON is only ever written on the single-threshold path (unchanged;
         --threshold-sweep still exits before any JSON write, exactly as
         before this patch).

Does NOT touch _score_windows() itself, the conformal calibration path,
run_at_threshold(), the collapse diagnostic, or any existing CLI flag --
this is a pure addition alongside the existing ratio/y_eval computation.

Run from ~/dissertation/ with akida_env activated:
    python3 apply_auprc_diagnostic_patch.py

Hard-refuses (exits nonzero, writes nothing) if any anchor text isn't found
exactly once -- re-run after checking whether the file changed since this
patch was written against the 4 July 2026 snapshot (post-force-scaler-
mismatch patch, the second patch to this file).
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


# ── 1. New import ─────────────────────────────────────────────────────────
old_1 = '''sys.path.insert(0, '.')
from src.evaluation.sliding_vote import event_level_metrics, collapse_diagnostic
from src.manifest import load_manifest, require_scaler_match'''

new_1 = '''sys.path.insert(0, '.')
from src.evaluation.sliding_vote import event_level_metrics, collapse_diagnostic
from src.manifest import load_manifest, require_scaler_match
from sklearn.metrics import average_precision_score'''

# ── 2. Compute + print AUPRC once, upstream of conformal/sweep/single-run ──
old_2 = '''ratio = _score_windows(X_eval, label='eval/test set')

# ── Candidate E: split-conformal threshold selection (sec4) ───────────────────'''

new_2 = '''ratio = _score_windows(X_eval, label='eval/test set')

# ── Action item 4: window-level AUPRC diagnostic (threshold-independent) ────
# Pure re-scoring of the existing continuous score against ground truth --
# no new inference. Computed once, before any threshold is chosen (spike-
# threshold, sweep, or conformal), so it's identical across all three modes
# for a given checkpoint/patient pair. Purpose: distinguish "genuinely
# inert" (poor ranking -- low AUPRC) from "threshold misplaced" (ranking is
# fine, the chosen cut point isn't) for chb13/chb15/chb16 specifically
# (Handoff_items1to3_done_item4_next.md sec4).
window_auprc = float(average_precision_score(y_eval, ratio))
print(f"\\n[Item 4] Window-level AUPRC (threshold-independent): {window_auprc:.4f}")

# ── Candidate E: split-conformal threshold selection (sec4) ───────────────────'''

# ── 3. Record AUPRC in the output JSON's window_level block ─────────────────
old_3 = '''        'window_level': {
            'sensitivity': round(win_sens, 4) if win_sens is not None else None,
            'specificity': round(win_spec, 4) if win_spec is not None else None,
            'fpr_per_hour': round(win_fpr_hr, 2) if win_fpr_hr is not None else None,
        },'''

new_3 = '''        'window_level': {
            'sensitivity': round(win_sens, 4) if win_sens is not None else None,
            'specificity': round(win_spec, 4) if win_spec is not None else None,
            'fpr_per_hour': round(win_fpr_hr, 2) if win_fpr_hr is not None else None,
            'auprc': round(window_auprc, 4),
        },'''


if __name__ == '__main__':
    patch_file(PATH, [
        (old_1, new_1),
        (old_2, new_2),
        (old_3, new_3),
    ])
    print("\nAll patches applied successfully.")
    print("\nSanity check:")
    print("  python3 -m py_compile src/evaluation/eval_event_level.py")
    print("\nRe-scoring targets (item 4 scope -- chb13/chb15/chb16, existing")
    print("C1 and C2 checkpoints, no new training):")
    print("  python3 src/evaluation/eval_event_level.py \\\\")
    print("      --fbz results/seizure_model_multi_noz_v2_w4a4.fbz \\\\")
    print("      --eval-patient chb13")
    print("  python3 src/evaluation/eval_event_level.py \\\\")
    print("      --fbz results/best_ann_chb13ft_frac0.25_v2_w4a4.fbz \\\\")
    print("      --eval-patient chb13")
    print("  (repeat for chb15, chb16, and both C1/C2 checkpoints of each)")
    print("\nRead AUPRC alongside Gate 0c collapse, not instead of it: a low")
    print("AUPRC with a PASS collapse still means the ranking itself is")
    print("weak; a reasonable AUPRC with a collapse FAIL means the ranking")
    print("is usable but the operating point picked was wrong -- these two")
    print("readings call for different next moves (retrain vs. re-threshold).")
