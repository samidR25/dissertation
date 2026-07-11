#!/usr/bin/env python3
"""
apply_lopo_eval_batching_patch.py
======================================
Fixes an OOM kill hit during the LOPO sweep, 10 July 2026: eval_event_level.py
called `akida_model.predict(X)` on the ENTIRE eval array in a single call.
The Akida CPU simulator's per-window memory cost scales with batch size
(internal spike-train buffers held across the whole call), so this worked
for chb10's 180,059-window full-recording eval set but got `Killed` (OOM)
for chb06 (240,228), chb07 (241,369), and chb09 (244,319) -- all comfortably
over 200k windows. Not a data problem, not a disk problem -- purely that the
single predict() call's peak memory scales with N and nobody had audited
this path against LOPO-sized (full-recording) eval sets before, only the
old ~30-40k-window per-patient chronological test slices.

Fix: batch the predict() call in fixed-size chunks, accumulate results.
This is inside `_score_windows()`, which BOTH the main eval pass and the
--conformal calibration pass go through (shared scoring pipeline, by
design -- see that function's own docstring), so one change covers both
call sites automatically.

Run from ~/dissertation/ with akida_env activated:
    python3 apply_lopo_eval_batching_patch.py

Hard-refuses (exits nonzero, writes nothing) if any anchor text isn't found
exactly once -- in particular, this patch assumes apply_lopo_full_eval_patch.py
was already applied (it anchors on the same --longctx line that patch
inserted before, which that patch does not modify).
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


# ── 1. New flag ──────────────────────────────────────────────────────────────
old_1 = """parser.add_argument('--longctx',        action='store_true')"""

new_1 = """parser.add_argument('--longctx',        action='store_true')
parser.add_argument('--eval-batch-size', type=int, default=8000,
                    help="Fix, 10 July 2026 LOPO session: windows per "
                         "akida_model.predict() call. The simulator's peak "
                         "memory scales with batch size, so an unbatched "
                         "call on a full-recording eval set (200k+ windows) "
                         "OOM-killed the process for several patients. "
                         "Default 8000 keeps peak memory well bounded "
                         "regardless of eval-set size; lower it further if "
                         "a future patient still OOMs.")"""

# ── 2. Batched predict() call inside _score_windows ─────────────────────────
old_2 = """    print(f"\\nRunning SNN simulator inference ({label})...")
    preds_raw = akida_model.predict(X)
    spike_counts = preds_raw.squeeze()  # (N,1,1,C) -> (N,C)"""

new_2 = """    print(f"\\nRunning SNN simulator inference ({label})...")
    # Batched (fix, 10 July 2026 LOPO session) -- see module docstring of
    # apply_lopo_eval_batching_patch.py for why this was necessary.
    _batch = args.eval_batch_size
    _preds_parts = []
    for _start in range(0, len(X), _batch):
        _preds_parts.append(akida_model.predict(X[_start:_start + _batch]))
    preds_raw = np.concatenate(_preds_parts, axis=0)
    del _preds_parts
    spike_counts = preds_raw.squeeze()  # (N,1,1,C) -> (N,C)"""


if __name__ == '__main__':
    patch_file(PATH, [
        (old_1, new_1),
        (old_2, new_2),
    ])
    print("\nAll patches applied successfully.")
    print("\nSanity check:")
    print("  python3 -m py_compile src/evaluation/eval_event_level.py")
