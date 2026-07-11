#!/usr/bin/env python3
"""
apply_lopo_pool_tag_patch.py
=================================
Supervisor-directed LOPO session, 9 July 2026: adds a --pool-tag flag to
train_baseline.py so each LOPO fold trains from its own fold-specific pooled
dataset (data/processed/multi_lopo_<patient>_dataset_ann.npz, produced by
build_dataset_lopo_fold.py) into its own uniquely-tagged checkpoint
(results/best_ann_multi_lopo_<patient>_v2.h5) -- no shared-output-path
collision risk, same discipline as every other tagged variant in this file
(--variant, --dann, --coral, etc.).

Two narrow changes, both in the "derive data_path / patient_tag" section:
  1. --multi-patient's data_path becomes fold-aware when --pool-tag is set.
  2. --multi-patient's patient_tag becomes fold-aware when --pool-tag is set.

Everything downstream (checkpoint path, manifest scaler-path resolution in
_write_ckpt_manifest, results JSON path) already derives from patient_tag --
confirmed the existing else-branch in _write_ckpt_manifest
(`f'data/processed/{patient_tag}_scaler.json'`) already resolves correctly
for a tag like 'multi_lopo_chb10' with NO further changes needed there, since
that tag matches neither 'multi' nor 'multi_from_'/'multi_sslpretrain'.

Does NOT touch the fine-tuning path, DANN/CORAL, or any existing tag's
behaviour -- --pool-tag defaults to None (unchanged behaviour when omitted).

Run from ~/dissertation/ with akida_env activated:
    python3 apply_lopo_pool_tag_patch.py

Hard-refuses (exits nonzero, writes nothing) if any anchor text isn't found
exactly once.
"""
import sys

PATH = 'src/models/train_baseline.py'


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
old_1 = '''parser.add_argument('--multi-patient',    action='store_true',
                    help="Train on pooled multi-patient dataset (multi_dataset_ann.npz)")'''

new_1 = '''parser.add_argument('--multi-patient',    action='store_true',
                    help="Train on pooled multi-patient dataset (multi_dataset_ann.npz)")
parser.add_argument('--pool-tag', default=None,
                    help="LOPO fold tag (supervisor-directed session, 9 July "
                         "2026). Requires --multi-patient. When set, loads "
                         "data/processed/multi_<pool-tag>_dataset_ann.npz "
                         "instead of the fixed 3-patient multi_dataset_ann.npz, "
                         "and tags all outputs 'multi_<pool-tag>' instead of "
                         "'multi' -- e.g. --pool-tag lopo_chb10 trains from "
                         "build_dataset_lopo_fold.py's chb10-held-out pool "
                         "and saves results/best_ann_multi_lopo_chb10_v2.h5. "
                         "Default None = unchanged existing behaviour.")'''

# ── 2. Fold-aware patient_tag ────────────────────────────────────────────────
old_2 = '''if args.multi_patient and args.finetune_from:
    patient_tag = f'multi_from_{args.finetune_from}'
elif args.multi_patient:
    patient_tag = 'multi'
else:
    patient_tag = args.patient'''

new_2 = '''if args.pool_tag and not args.multi_patient:
    parser.error('--pool-tag requires --multi-patient')
if args.multi_patient and args.finetune_from:
    patient_tag = f'multi_from_{args.finetune_from}'
elif args.multi_patient:
    patient_tag = f'multi_{args.pool_tag}' if args.pool_tag else 'multi'
else:
    patient_tag = args.patient'''

# ── 3. Fold-aware data_path ──────────────────────────────────────────────────
# NOTE: triple-DOUBLE-quotes here deliberately -- the anchor's last line ends
# in a single-quote (f'...npz'), which would collide with a triple-single-
# quote closer (4 quotes in a row = SyntaxError, caught the hard way).
old_3 = """if args.multi_patient:
    data_path = 'data/processed/multi_dataset_ann.npz'
else:
    data_path = f'data/processed/{args.patient}_dataset_ann.npz'"""

new_3 = """if args.multi_patient and args.pool_tag:
    data_path = f'data/processed/multi_{args.pool_tag}_dataset_ann.npz'
elif args.multi_patient:
    data_path = 'data/processed/multi_dataset_ann.npz'
else:
    data_path = f'data/processed/{args.patient}_dataset_ann.npz'"""


if __name__ == '__main__':
    patch_file(PATH, [
        (old_1, new_1),
        (old_2, new_2),
        (old_3, new_3),
    ])
    print("\nAll patches applied successfully.")
    print("\nSanity check:")
    print("  python3 -m py_compile src/models/train_baseline.py")
    print("\nLOPO fold training, once build_dataset_lopo_fold.py has produced "
          "data/processed/multi_lopo_chb10_dataset_ann.npz:")
    print("  python3 src/models/train_baseline.py --model-version 2 "
          "--multi-patient --pool-tag lopo_chb10")
