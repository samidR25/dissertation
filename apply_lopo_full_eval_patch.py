#!/usr/bin/env python3
"""
apply_lopo_full_eval_patch.py
==================================
Supervisor-directed LOPO session, 9 July 2026: adds a --lopo-full flag to
eval_event_level.py so a LOPO fold's held-out patient is evaluated on their
FULL recording (data/processed/<patient>_dataset_lopo_full.npz, produced by
build_lopo_eval_set.py) instead of the old 15% chronological test slice --
matching standard LOPO methodology and the Ali et al. (2024) comparator
directly (agreed decision: "Full recording, matches Ali et al. directly").

One change, in the data_path resolution block. Everything downstream
(scoring, event-level metrics, collapse diagnostic, Gate 1b scaler check,
output JSON) is UNCHANGED -- Gate 1b already correctly no-ops for a held-out
patient that was never a pool constituent (confirmed: the existing
'eval_patient not in per_patient map' branch already prints the correct
"normal cross-patient generalisation case" note and skips the check).

Mutually exclusive with --longctx/--g-features (same pattern already used
between those two).

Run from ~/dissertation/ with akida_env activated:
    python3 apply_lopo_full_eval_patch.py

Hard-refuses (exits nonzero, writes nothing) if any anchor text isn't found
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


# ── 1. New flag ──────────────────────────────────────────────────────────────
old_1 = '''parser.add_argument('--longctx',        action='store_true')'''

new_1 = '''parser.add_argument('--lopo-full',      action='store_true',
                    help="Supervisor-directed LOPO session, 9 July 2026: "
                         "evaluate on the held-out patient's FULL recording "
                         "(data/processed/<patient>_dataset_lopo_full.npz, "
                         "from build_lopo_eval_set.py) instead of the "
                         "chronological 15% test slice. Matches standard "
                         "LOPO methodology and the Ali et al. (2024) "
                         "comparator directly. Mutually exclusive with "
                         "--longctx/--g-features.")
parser.add_argument('--longctx',        action='store_true')'''

# ── 2. Fold-aware data_path ──────────────────────────────────────────────────
old_2 = '''if args.longctx and args.g_features:
    sys.exit("ERROR: --longctx and --g-features are mutually exclusive.")
if args.longctx:
    data_path = (f'data/processed/{args.eval_patient}_dataset_longctx_w'
                 f'{args.window_samples}.npz')
elif args.g_features:
    data_path = f'data/processed/{args.eval_patient}_dataset_g.npz'
else:
    data_path = f'data/processed/{args.eval_patient}_dataset_ann.npz'
assert os.path.exists(data_path), f"Dataset not found: {data_path}"'''

new_2 = '''if sum([args.longctx, args.g_features, args.lopo_full]) > 1:
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
    + ("\\nRun: python3 src/preprocessing/build_lopo_eval_set.py "
       f"--patient {args.eval_patient}" if args.lopo_full else ""))'''


if __name__ == '__main__':
    patch_file(PATH, [
        (old_1, new_1),
        (old_2, new_2),
    ])
    print("\nAll patches applied successfully.")
    print("\nSanity check:")
    print("  python3 -m py_compile src/evaluation/eval_event_level.py")
    print("\nLOPO fold eval, once build_lopo_eval_set.py has produced "
          "data/processed/chb10_dataset_lopo_full.npz:")
    print("  python3 src/evaluation/eval_event_level.py \\\\")
    print("      --fbz results/seizure_model_multi_lopo_chb10_v2_w4a4.fbz \\\\")
    print("      --eval-patient chb10 --lopo-full")
