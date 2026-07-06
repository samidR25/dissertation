"""
Build train/validation/test splits from preprocessed CHB-MIT data.
Split strategy: CHRONOLOGICAL (not random).
Why chronological?
  CHB-MIT windows are created with 50% overlap. A random split would put
  windows from the same seizure event (or even the same 2s segment) into
  both train and test, causing data leakage. Chronological splitting
  ensures the model is evaluated on data it has never seen in any form.
  This also better simulates real deployment: the model is trained on
  historical recordings and tested on future ones.
Split: 70% train / 15% val / 15% test (chronological)
Imbalance handling: SMOTE on training set only.
  Why training set only?
    Applying SMOTE to val/test would create synthetic seizure examples in
    evaluation, making results unrealistically optimistic. Evaluate on
    real data distributions only.
Input scaling:
  AKD1000 InputConvolutional expects uint8-range inputs [0, 255].
  Raw CHB-MIT EEG is float32 ~[-0.0016, 0.0016] (microvolts).
  Scale parameters computed from X_train ONLY and saved to JSON.
  All downstream scripts (train, convert, hardware) load this JSON —
  no per-script scaling logic anywhere.

References:
  SMOTE: Chawla et al. (2002). JAIR 16, 321-357.
  Chronological splitting: Gemein et al. (2020). NeuroImage.

Usage:
    python3 src/preprocessing/build_dataset.py --patient chb01
    python3 src/preprocessing/build_dataset.py --patient chb02
"""
import argparse
import json
import os
import numpy as np
from imblearn.over_sampling import SMOTE


def chronological_split(X, y, train_frac=0.70, val_frac=0.15):
    """
    Split arrays chronologically.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))
    return (X[:train_end],        y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:],          y[val_end:])


def undersample_train(X_train, y_train, ratio=10, random_state=42):
    """
    Undersample majority (non-seizure) class to `ratio`:1 vs seizure count.
    Returns the REAL, pre-SMOTE undersampled (X_sub, y_sub) — no synthetic data.
    This is the array that must be split BEFORE any SMOTE call, ever.
    """
    seizure_idx    = np.where(y_train == 1)[0]
    nonseizure_idx = np.where(y_train == 0)[0]
    n_seizure = len(seizure_idx)

    n_keep = min(len(nonseizure_idx), n_seizure * ratio)
    rng = np.random.default_rng(random_state)
    kept_idx = rng.choice(nonseizure_idx, size=n_keep, replace=False)
    combined_idx = np.sort(np.concatenate([seizure_idx, kept_idx]))

    X_sub = X_train[combined_idx]
    y_sub = y_train[combined_idx]
    print(f"  Undersampled (real, pre-SMOTE): {y_sub.sum()} seizure, "
          f"{(y_sub==0).sum()} non-seizure")
    return X_sub, y_sub


def smote_oversample(X_sub, y_sub, random_state=42):
    """
    SMOTE-balance an already-undersampled REAL set. Does no splitting —
    caller is responsible for ensuring X_sub/y_sub contain no cross-split
    contamination before this is called.
    """
    n, ch, t = X_sub.shape
    n_seiz = int(y_sub.sum())
    X_flat = X_sub.reshape(len(X_sub), -1)
    sm = SMOTE(random_state=random_state, k_neighbors=min(5, n_seiz - 1))
    X_res, y_res = sm.fit_resample(X_flat, y_sub)
    X_res = X_res.reshape(-1, ch, t)
    print(f"  After SMOTE:  {y_res.sum()} seizure, "
          f"{(y_res==0).sum()} non-seizure")
    return X_res, y_res


def apply_smote(X_train, y_train, random_state=42):
    """Back-compat wrapper: undersample then SMOTE in one call (original behaviour)."""
    X_sub, y_sub = undersample_train(X_train, y_train, ratio=10, random_state=random_state)
    return smote_oversample(X_sub, y_sub, random_state=random_state)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', default='chb01',
                        help='Patient ID (e.g. chb01, chb02)')
    parser.add_argument('--output-dir', default='data/processed/')
    parser.add_argument('--train-frac', type=float, default=0.70)
    parser.add_argument('--val-frac',   type=float, default=0.15)
    args = parser.parse_args()

    pid = args.patient
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    X_path = os.path.join(out, f'{pid}_X.npy')
    y_path = os.path.join(out, f'{pid}_y.npy')
    assert os.path.exists(X_path) and os.path.exists(y_path), \
        f"Raw windows not found: {X_path}\n" \
        f"Run first: python3 src/preprocessing/preprocess.py --patient {pid}"

    X = np.load(X_path)
    y = np.load(y_path)
    print(f"\nPatient: {pid}")
    print(f"Loaded: {X.shape}  |  {y.sum()} seizure windows  "
          f"({100*y.mean():.2f}%)")

    # ── Chronological split ───────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = \
        chronological_split(X, y, args.train_frac, args.val_frac)

    print(f"\nSplit (chronological {int(args.train_frac*100)}/"
          f"{int(args.val_frac*100)}/{int((1-args.train_frac-args.val_frac)*100)}):")
    print(f"  Train : {X_train.shape}  seizures: {y_train.sum()}")
    print(f"  Val   : {X_val.shape}    seizures: {y_val.sum()}")
    print(f"  Test  : {X_test.shape}   seizures: {y_test.sum()}")
# ── SMOTE on training set only ────────────────────────────────────────────
    if y_train.sum() == 0:
        print(f"\nWARNING: {pid} training split has ZERO seizure windows — "
              "inverse of chb01/02/05's pattern. All seizures fall in val/test. "
              "Skipping SMOTE (no minority class to balance). This patient "
              "cannot be trained standalone, but the saved npz is still valid "
              "for held-out evaluation (X_test).")
        X_train_bal, y_train_bal   = X_train, y_train
        X_train_real, y_train_real = X_train, y_train   # nothing to undersample
    else:
        print("\nApplying SMOTE to training set...")
        X_train_real, y_train_real = undersample_train(X_train, y_train)
        X_train_bal,  y_train_bal  = smote_oversample(X_train_real, y_train_real)
# ── Input scaling: [0, 255] ───────────────────────────────────────────────
    # Scale parameters from X_train ONLY. Saved to JSON for reference.
    # The Rescaling layer in akida_cnn.py divides by 255 internally so the
    # network sees [0,1] while the quantiser sees [0,255] at the input.
    # Single dataset format used for training, conversion, and hardware inference.
    X_min  = float(X_train_bal.min())
    X_max  = float(X_train_bal.max())
    scale  = 255.0 / (X_max - X_min)
    shift  = -X_min * scale
    scaler = {'scale': scale, 'shift': shift,
              'X_min': X_min, 'X_max': X_max}

    scaler_path = os.path.join(out, f'{pid}_scaler.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler, f, indent=2)
    print(f"\nScaler saved: {scaler_path}")
    print(f"  Raw range: [{X_min:.6f}, {X_max:.6f}]")
    print(f"  scale={scale:.2f}  shift={shift:.2f}")

    def scale_255(X):
        return np.clip(X.astype('float32') * scale + shift, 0.0, 255.0)
# ── Save ANN dataset ([0,255]) ────────────────────────────────────────────
    ann_path = os.path.join(out, f'{pid}_dataset_ann.npz')
    if y_train.sum() == 0:
        print(f"\nNOTE: {pid} has zero train-split seizures — skipping "
              "X_train/y_train (and X_train_real/y_train_real) in the saved "
              "npz entirely (unbalanced, unusable for training, and large "
              "enough to risk OOM on WSL2). Held-out eval (X_test/X_val) is "
              "unaffected.")
        X_train_save = np.empty((0, *X_train_bal.shape[1:]), dtype='float32')
        y_train_save = np.empty((0,), dtype=y_train_bal.dtype)
        X_train_real_save = np.empty((0, *X_train_real.shape[1:]), dtype='float32')
        y_train_real_save = np.empty((0,), dtype=y_train_real.dtype)
    else:
        X_train_save = scale_255(X_train_bal)
        y_train_save = y_train_bal
        X_train_real_save = scale_255(X_train_real)
        y_train_real_save = y_train_real
    np.savez_compressed(
        ann_path,
        X_train=X_train_save, y_train=y_train_save,
        X_train_real=X_train_real_save, y_train_real=y_train_real_save,
        X_val=scale_255(X_val),         y_val=y_val,
        X_test=scale_255(X_test),       y_test=y_test,)
    print(f"ANN dataset saved: {ann_path}")
# ── Gate 1b: manifest sidecar for this dataset artifact ──────────────────
    import sys as _sys
    _sys.path.insert(0, '.')
    from src.manifest import write_manifest
    write_manifest(
        ann_path,
        patient=pid,
        scaler_path=scaler_path,
        scaler=scaler,
        split_ratios=[args.train_frac, args.val_frac,
                      round(1 - args.train_frac - args.val_frac, 4)],
        split_counts={'train': int(len(X_train)), 'val': int(len(X_val)),
                      'test': int(len(X_test))},
        smote_applied=bool(y_train.sum() != 0),
        smote_undersample_ratio=10,
        zscore=False,
    )
    print(f"Manifest saved: {ann_path}.manifest.json")
    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Dataset summary — {pid} (record in dissertation Table 1)")
    print(f"{'='*55}")
    print(f"  Total windows    : {len(X)}")
    print(f"  Window size      : 2.0s @ 256Hz = 512 samples")
    print(f"  Channels         : {X.shape[1]}")
    print(f"  Overlap          : 50%")
    print(f"  Split            : chronological 70/15/15")
    print(f"  Train (balanced) : {len(X_train_bal)} "
          f"({y_train_bal.sum()} seizure / "
          f"{(y_train_bal==0).sum()} non-seizure)")
    print(f"  Val  (real dist) : {len(X_val)} ({y_val.sum()} seizure)")
    print(f"  Test (real dist) : {len(X_test)} ({y_test.sum()} seizure)")
    print(f"  Input format     : [0.0, 255.0] float32")
    print(f"  Rescaling        : handled by akida_cnn.py Rescaling layer")
    print(f"  Scaler ref       : data/processed/{pid}_scaler.json")
    print(f"{'='*55}")
    print(f"\nNext: python3 src/models/train_baseline.py --patient {pid}")
