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


def apply_smote(X_train, y_train, random_state=42):
    """
    Undersample majority class then SMOTE minority to balance.
    Full SMOTE on 100k+ windows exceeds WSL2 memory — undersample first
    to a manageable ratio, then oversample seizure class to match.
    """
    n, ch, t = X_train.shape
    seizure_idx    = np.where(y_train == 1)[0]
    nonseizure_idx = np.where(y_train == 0)[0]
    n_seizure = len(seizure_idx)

    # Keep 10x non-seizure windows before SMOTE to manage memory
    n_keep = min(len(nonseizure_idx), n_seizure * 10)
    rng = np.random.default_rng(random_state)
    kept_idx = rng.choice(nonseizure_idx, size=n_keep, replace=False)
    combined_idx = np.sort(np.concatenate([seizure_idx, kept_idx]))

    X_sub = X_train[combined_idx]
    y_sub = y_train[combined_idx]
    print(f"  Before SMOTE (after undersample): {y_sub.sum()} seizure, "
          f"{(y_sub==0).sum()} non-seizure")

    X_flat = X_sub.reshape(len(X_sub), -1)
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_flat, y_sub)
    X_res = X_res.reshape(-1, ch, t)
    print(f"  After SMOTE:  {y_res.sum()} seizure, "
          f"{(y_res==0).sum()} non-seizure")
    return X_res, y_res


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

    # ── Load raw windows ──────────────────────────────────────────────────────
    windows_path = os.path.join(out, f'{pid}_windows.npz')
    assert os.path.exists(windows_path), \
        f"Raw windows not found: {windows_path}\n" \
        f"Run first: python3 src/preprocessing/preprocess.py --patient {pid}"

    data = np.load(windows_path)
    X, y = data['X'], data['y']
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
    print("\nApplying SMOTE to training set...")
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

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
    np.savez_compressed(
        ann_path,
        X_train=scale_255(X_train_bal), y_train=y_train_bal,
        X_val=scale_255(X_val),         y_val=y_val,
        X_test=scale_255(X_test),       y_test=y_test
    )
    print(f"ANN dataset saved: {ann_path}")

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
