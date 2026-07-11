#!/usr/bin/env python3
"""
apply_lopo_eval_loading_patch.py
=====================================
Second, independent fix for the same OOM pattern apply_lopo_eval_batching_
patch.py addressed. That patch bounded akida_model.predict()'s peak memory
by batching it -- necessary, but not sufficient, because the line that
loads X_eval in the first place ALREADY makes an unnecessary full duplicate
copy of the entire eval array before batching ever gets a chance to help:

    X_eval = data['X_test'][..., np.newaxis].astype('float32')

`.astype()` copies by default even when the source dtype already matches
the target (float32 here -- build_lopo_eval_set.py always saves X_test as
float32). For a ~240k-window full recording that's two ~8.8GB arrays alive
at once (the decompressed npz array, then the astype duplicate) before
inference even starts. Same category of bug as the np.asarray-vs-np.array
issue fixed earlier in build_lopo_eval_set.py, opposite direction: there we
needed to FORCE a copy (to get a writable array off a read-only memmap
view); here we want to AVOID an unnecessary one.

Fix: `.astype('float32', copy=False)` -- returns the same array with no
copy when the dtype already matches, copies only when it actually needs to
(e.g. for --longctx/--g-features paths or older non-lopo datasets that
might not be float32). Behaviourally identical either way; just doesn't
double memory for the common case.

Run from ~/dissertation/ with akida_env activated, AFTER
apply_lopo_eval_batching_patch.py:
    python3 apply_lopo_eval_loading_patch.py

Hard-refuses (exits nonzero, writes nothing) if the anchor text isn't found
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


old_1 = """if args.longctx or args.g_features:
    X_eval = data['X_test'].astype('float32')
else:
    X_eval = data['X_test'][..., np.newaxis].astype('float32')"""

new_1 = """if args.longctx or args.g_features:
    X_eval = data['X_test'].astype('float32', copy=False)
else:
    X_eval = data['X_test'][..., np.newaxis].astype('float32', copy=False)"""


if __name__ == '__main__':
    patch_file(PATH, [(old_1, new_1)])
    print("\nPatch applied successfully.")
    print("\nSanity check:")
    print("  python3 -m py_compile src/evaluation/eval_event_level.py")
