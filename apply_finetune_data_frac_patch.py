#!/usr/bin/env python3
"""
apply_finetune_data_frac_patch.py
============================================
C2 compounding, action item 3 (Handoff_g_closed_to_c2_compounding.md sec5):
data-efficiency sweep to actually measure the dissertation's "adapts with
minimal labelled data" claim, currently asserted, not measured.

Confirmed by reading train_baseline.py this session (line numbers below
are against the 4 July 2026 snapshot -- re-check if this drifts):

  - The --freeze-depth fine-tuning path and fresh multi-patient training
    both fall through to the SAME shared call, _stratified_real_split()
    (the `else` branch, ~line 721-727) -- this is Gate 2d's generalisation
    of the Gate 0a leak-free-split pattern to cover both cases at once.
  - Subsampling needs to happen INSIDE _stratified_real_split(), before
    its own train_test_split/SMOTE calls, not duplicated at the call site
    -- keeps the leak-free-split logic in one place.
  - Subsampling must be stratified (preserve the seizure/non-seizure
    ratio) and must ONLY apply when --finetune-from is set -- fresh
    baseline/pool training must never be silently subsampled by this flag.

Applies three edits to src/models/train_baseline.py:

  1. New --finetune-data-frac flag (default 1.0 = unchanged behaviour).
     Refused if given without --finetune-from, matching this project's
     existing refuse-don't-silently-noop pattern (--gradual-unfreeze
     already refuses without --finetune-from at the same validation block).
  2. _stratified_real_split() gains a `data_frac` parameter. When < 1.0,
     stratified-subsamples X_train_real/y_train_real (via train_test_split
     itself, reusing sklearn's own stratify machinery rather than hand-
     rolling a second sampling method) BEFORE the existing 80/20 val split
     and BEFORE SMOTE -- so the fraction claim is about real, labelled
     windows actually used, not post-SMOTE-inflated counts.
  3. The shared call site passes data_frac=args.finetune_data_frac only
     when args.finetune_from is set; fresh training always gets data_frac=1.0
     regardless of the flag's value (defence in depth alongside the
     argparse-level refusal in edit 1).

Does NOT touch build_dataset.py, SMOTE's own logic, or the gradual_unfreeze/
stft branches (out of scope for item 3 -- freeze-depth is this project's
established fine-tuning path, per the handoff's own scoping).

Run from ~/dissertation/ with akida_env activated:
    python3 apply_finetune_data_frac_patch.py

Hard-refuses (exits nonzero, writes nothing) if any anchor text isn't found
exactly once -- re-run after checking whether the file changed since this
patch was written.
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


# ── 1. _stratified_real_split gains a data_frac parameter ───────────────────
old_1 = '''def _stratified_real_split(data, seed=42, test_size=0.20, already_multichannel=False):
    """
    Gate 0a pattern (originally built for --gradual-unfreeze only),
    generalised in Gate 2d to cover fresh training and --freeze-depth too.

    Splits the REAL, undersampled, pre-SMOTE train pool (X_train_real/
    y_train_real) stratified, so val is GUARANTEED to contain seizure
    windows regardless of how they fall chronologically. SMOTE applied
    only to the resulting train portion. Val is real, untouched — used
    ONLY for early-stopping/checkpoint selection, never reported as a
    held-out metric (that's still the chronological X_test).

    chb03's chronological val has ZERO seizures (all 3 fall in train/test
    by accident) — any path relying on the raw chronological val for
    ModelCheckpoint(monitor='val_loss') silently selects a checkpoint
    based on nothing but false-positive noise. chb10 never exposed this
    because its chronological val happened to contain 66 seizures.

    already_multichannel: True for --stft/--longctx data, which is already
    (N, 18, window_samples, 3) — applying [..., np.newaxis] to that would
    produce an incorrect 5D array. False (default) for the 1-channel
    baseline, which needs the newaxis to become (N, 18, window_samples, 1).
    """
    if 'X_train_real' not in data.files:
        sys.exit(
            "ERROR: dataset predates the Gate 0a fix (missing "
            "X_train_real/y_train_real). Regenerate via:\\n"
            "  python3 src/preprocessing/build_dataset.py --patient <pid>"
        )
    if already_multichannel:
        X_train_real = data['X_train_real'].astype('float32')
    else:
        X_train_real = data['X_train_real'][..., np.newaxis].astype('float32')
    y_train_real = data['y_train_real']

    n_seiz = int(y_train_real.sum())
    if n_seiz < 2:
        sys.exit(f"ERROR: only {n_seiz} seizure window(s) in the real "
                 "train pool — cannot stratified-split for a seizure-"
                 "guaranteed val.")

    X_tr_real, X_vl, y_tr_real, y_vl = train_test_split(
        X_train_real, y_train_real, test_size=test_size,
        stratify=y_train_real, random_state=seed,
    )'''

new_1 = '''def _stratified_real_split(data, seed=42, test_size=0.20,
                            already_multichannel=False, data_frac=1.0):
    """
    Gate 0a pattern (originally built for --gradual-unfreeze only),
    generalised in Gate 2d to cover fresh training and --freeze-depth too.

    Splits the REAL, undersampled, pre-SMOTE train pool (X_train_real/
    y_train_real) stratified, so val is GUARANTEED to contain seizure
    windows regardless of how they fall chronologically. SMOTE applied
    only to the resulting train portion. Val is real, untouched — used
    ONLY for early-stopping/checkpoint selection, never reported as a
    held-out metric (that's still the chronological X_test).

    chb03's chronological val has ZERO seizures (all 3 fall in train/test
    by accident) — any path relying on the raw chronological val for
    ModelCheckpoint(monitor='val_loss') silently selects a checkpoint
    based on nothing but false-positive noise. chb10 never exposed this
    because its chronological val happened to contain 66 seizures.

    already_multichannel: True for --stft/--longctx data, which is already
    (N, 18, window_samples, 3) — applying [..., np.newaxis] to that would
    produce an incorrect 5D array. False (default) for the 1-channel
    baseline, which needs the newaxis to become (N, 18, window_samples, 1).

    data_frac: C2 compounding item 3 (data-efficiency sweep). When < 1.0,
    stratified-subsamples X_train_real/y_train_real to this fraction BEFORE
    the val split and BEFORE SMOTE, so the fraction is about real, labelled
    windows actually available for fine-tuning, not post-SMOTE-inflated
    counts. Default 1.0 = unchanged behaviour (no subsampling).
    """
    if 'X_train_real' not in data.files:
        sys.exit(
            "ERROR: dataset predates the Gate 0a fix (missing "
            "X_train_real/y_train_real). Regenerate via:\\n"
            "  python3 src/preprocessing/build_dataset.py --patient <pid>"
        )
    if already_multichannel:
        X_train_real = data['X_train_real'].astype('float32')
    else:
        X_train_real = data['X_train_real'][..., np.newaxis].astype('float32')
    y_train_real = data['y_train_real']

    if data_frac < 1.0:
        n_seiz_full = int(y_train_real.sum())
        if n_seiz_full < 2:
            sys.exit(f"ERROR: only {n_seiz_full} seizure window(s) in the "
                     "real train pool BEFORE subsampling — cannot "
                     "stratified-subsample.")
        X_train_real, _, y_train_real, _ = train_test_split(
            X_train_real, y_train_real, train_size=data_frac,
            stratify=y_train_real, random_state=seed,
        )
        print(f"\\n  [Item 3: data-efficiency sweep] Subsampled real train "
              f"pool to {data_frac:.0%} ({len(y_train_real)} windows, "
              f"{int(y_train_real.sum())} seizure) BEFORE val split/SMOTE.")

    n_seiz = int(y_train_real.sum())
    if n_seiz < 2:
        sys.exit(f"ERROR: only {n_seiz} seizure window(s) in the real "
                 "train pool — cannot stratified-split for a seizure-"
                 "guaranteed val.")

    X_tr_real, X_vl, y_tr_real, y_vl = train_test_split(
        X_train_real, y_train_real, test_size=test_size,
        stratify=y_train_real, random_state=seed,
    )'''

# ── 2. New CLI flag ───────────────────────────────────────────────────────────
old_2 = '''parser.add_argument('--freeze-depth', type=int, default=None,
                    help="Freeze the first N conv blocks' KERNELS during "
                         "fine-tuning (0=full fine-tune .. 3=head-only). "
                         "BatchNorm layers are always trainable regardless "
                         "of this value (Gate 2b). Mutually exclusive with "
                         "--gradual-unfreeze — if both are omitted, falls "
                         "back to head-only fine-tuning (Phase 2a behaviour).")'''

new_2 = '''parser.add_argument('--freeze-depth', type=int, default=None,
                    help="Freeze the first N conv blocks' KERNELS during "
                         "fine-tuning (0=full fine-tune .. 3=head-only). "
                         "BatchNorm layers are always trainable regardless "
                         "of this value (Gate 2b). Mutually exclusive with "
                         "--gradual-unfreeze — if both are omitted, falls "
                         "back to head-only fine-tuning (Phase 2a behaviour).")
parser.add_argument('--finetune-data-frac', type=float, default=1.0,
                    help="C2 compounding item 3 (data-efficiency sweep): "
                         "stratified-subsample the patient's real, "
                         "pre-SMOTE fine-tuning pool (X_train_real/"
                         "y_train_real) to this fraction (e.g. 0.25) "
                         "BEFORE the val split and BEFORE SMOTE. Default "
                         "1.0 = unchanged behaviour. Requires "
                         "--finetune-from; refused otherwise (fresh/pool "
                         "training must never be silently subsampled).")'''

# ── 3. Validation: refuse if used without --finetune-from ───────────────────
old_3 = '''if args.gradual_unfreeze and args.finetune_from is None:
    parser.error("--gradual-unfreeze requires --finetune-from")'''

new_3 = '''if args.gradual_unfreeze and args.finetune_from is None:
    parser.error("--gradual-unfreeze requires --finetune-from")
if args.finetune_data_frac < 1.0 and args.finetune_from is None:
    parser.error("--finetune-data-frac requires --finetune-from -- fresh/"
                 "pool training must never be silently subsampled by this "
                 "flag.")
if not (0.0 < args.finetune_data_frac <= 1.0):
    parser.error("--finetune-data-frac must be in (0.0, 1.0]")'''

# ── 4. Call site: pass data_frac only for the finetune-from case ────────────
old_4 = '''else:
    # Gate 2d: fresh training and --freeze-depth fine-tuning both fall
    # through to here — neither was protected by the Gate 0a fix before
    # now. Same seizure-guaranteed split as gradual_unfreeze already had.
    print(f"\\nDataset: {data_path}")
    X_train, y_train, X_val, y_val = _stratified_real_split(
        data, seed=args.seed, already_multichannel=(args.stft or args.longctx))'''

new_4 = '''else:
    # Gate 2d: fresh training and --freeze-depth fine-tuning both fall
    # through to here — neither was protected by the Gate 0a fix before
    # now. Same seizure-guaranteed split as gradual_unfreeze already had.
    print(f"\\nDataset: {data_path}")
    # Item 3: data_frac only ever applies to fine-tuning (argparse already
    # refuses --finetune-data-frac<1.0 without --finetune-from, this is
    # defence in depth so fresh/pool training is never subsampled even if
    # that check is ever bypassed or the flag's default changes upstream).
    _data_frac = args.finetune_data_frac if args.finetune_from else 1.0
    X_train, y_train, X_val, y_val = _stratified_real_split(
        data, seed=args.seed, already_multichannel=(args.stft or args.longctx),
        data_frac=_data_frac)'''


if __name__ == '__main__':
    patch_file(PATH, [
        (old_1, new_1),
        (old_2, new_2),
        (old_3, new_3),
        (old_4, new_4),
    ])
    print("\nAll patches applied successfully.")
    print("\nSanity check:")
    print("  python3 -m py_compile src/models/train_baseline.py")
    print("\nItem 3 sweep (per handoff sec5 scope: chb10 + chb13 only, "
          "single seed per fraction, full metric bundle each time):")
    print("  for p in chb10 chb13; do")
    print("    for frac in 0.10 0.25 0.50 1.00; do")
    print("      python3 src/models/train_baseline.py --patient $p "
          "--finetune-from multi --freeze-depth 2 --seed 256 "
          "--finetune-data-frac $frac --model-version 2")
    print("      mv results/best_ann_${p}_v2.h5 "
          "results/best_ann_${p}ft_frac${frac}_v2.h5")
    print("      python3 src/models/convert_to_snn.py "
          "--base ${p}ft_frac${frac} --patient $p --eval-patient $p "
          "--model-version 2 --w-bits 4 --a-bits 4 --cal-samples 256 --seed 1")
    print("      python3 src/evaluation/eval_event_level.py "
          "--fbz results/seizure_model_${p}ft_frac${frac}_v2_w4a4.fbz "
          "--eval-patient $p")
    print("    done")
    print("  done")
    print("\nReminder: report the FULL metric bundle (event sensitivity, "
          "FP/hr, Gate 0c collapse) at EACH fraction, not sensitivity "
          "alone -- a fraction that 'works' by collapsing into an "
          "over-firing block is not a real result.")
