"""
apply_roc_auc_patch.py
=========================
Adds ROC-AUC alongside the existing window-level AUPRC diagnostic in
eval_event_level.py. Same pattern as the AUPRC addition -- pure re-scoring
of the already-computed continuous `ratio` score against ground truth, no
new inference, no retraining, no reconversion. Requested by supervisor
for direct comparability with literature that reports ROC-AUC rather than
AUPRC (most of the field). Report ROC-AUC alongside AUPRC, not instead of
it -- ROC-AUC is known to look artificially strong under this project's
~1-3% seizure-window class imbalance (false positive RATE is diluted by
a huge negative-class denominator), which is exactly the kind of
optimism the collapse gate elsewhere in this pipeline exists to catch.
AUPRC should remain the metric the write-up leads with; ROC-AUC is an
additional, not a replacement, number.

Run from ~/dissertation/ with akida_env activated:
    python3 apply_roc_auc_patch.py

Hard-refuses (exits nonzero, writes nothing) if any anchor isn't found
exactly once.
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


# ── 1. Import ────────────────────────────────────────────────────────────────
old_1 = """from sklearn.metrics import average_precision_score"""

new_1 = """from sklearn.metrics import average_precision_score, roc_auc_score"""

# ── 2. Computation, right next to the existing AUPRC block ─────────────────────
old_2 = """window_auprc = float(average_precision_score(y_eval, ratio))
print(f"\\n[Item 4] Window-level AUPRC (threshold-independent): {window_auprc:.4f}")"""

new_2 = """window_auprc = float(average_precision_score(y_eval, ratio))
print(f"\\n[Item 4] Window-level AUPRC (threshold-independent): {window_auprc:.4f}")

# ── ROC-AUC, alongside AUPRC (supervisor request) ──────────────────────────
# Same re-scoring of `ratio` against y_eval, no new inference. Report
# alongside AUPRC, not instead of it -- ROC-AUC is known to look
# artificially strong under this project's severe class imbalance, since
# FPR is diluted by a large negative-class denominator in a way precision
# is not. Guarded for the single-class edge case (e.g. an eval set with
# zero seizure windows), where ROC-AUC is undefined.
try:
    window_roc_auc = float(roc_auc_score(y_eval, ratio))
    print(f"[Item 4b] Window-level ROC-AUC: {window_roc_auc:.4f}  "
          f"(report alongside AUPRC, not instead of it -- see docstring)")
except ValueError:
    window_roc_auc = None
    print("[Item 4b] Window-level ROC-AUC: undefined (only one class "
          "present in y_eval for this patient)")"""

# ── 3. JSON output, alongside the existing 'auprc' field ───────────────────────
old_3 = """            'auprc': round(window_auprc, 4),
        },"""

new_3 = """            'auprc': round(window_auprc, 4),
            'roc_auc': round(window_roc_auc, 4) if window_roc_auc is not None else None,
        },"""

if __name__ == '__main__':
    patch_file(PATH, [(old_1, new_1), (old_2, new_2), (old_3, new_3)])
