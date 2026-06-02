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
  - For chb01 with 17 sessions: sessions 1-11 train, 12-14 val, 15-17 test

Imbalance handling: SMOTE on training set only.
  Why training set only?
    Applying SMOTE to val/test would create synthetic seizure examples in
    evaluation, making results unrealistically optimistic. Evaluate on
    real data distributions only.

Reference:
  SMOTE: Chawla et al. (2002). JAIR 16, 321-357.
  Chronological splitting: Gemein et al. (2020). NeuroImage.
"""
import numpy as np
from imblearn.over_sampling import SMOTE
import os

def chronological_split(X, y, train_frac=0.70, val_frac=0.15):
    """
    Split arrays chronologically.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    return (X[:train_end],       y[:train_end],
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
    # Keep 10x non-seizure windows (still heavily imbalanced pre-SMOTE,
    # but reduces memory from ~7.5GB to ~450MB)
    n_keep = min(len(nonseizure_idx), n_seizure * 10)
    rng = np.random.default_rng(random_state)
    kept_idx = rng.choice(nonseizure_idx, size=n_keep, replace=False)

    combined_idx = np.concatenate([seizure_idx, kept_idx])
    combined_idx.sort()

    X_sub = X_train[combined_idx]
    y_sub = y_train[combined_idx]

    print(f"Before SMOTE (after undersample): {y_sub.sum()} seizure, "
          f"{(y_sub==0).sum()} non-seizure")

    X_flat = X_sub.reshape(len(X_sub), -1)
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_flat, y_sub)
    X_res = X_res.reshape(-1, ch, t)

    print(f"After SMOTE:  {y_res.sum()} seizure, "
          f"{(y_res==0).sum()} non-seizure")
    return X_res, y_res

if __name__ == '__main__':
    # ── Load raw windows (for ANN training) ───────────────────────────────────
    data = np.load('data/processed/chb01_windows.npz')
    X, y = data['X'], data['y']

    print(f"Loaded: {X.shape}  |  {y.sum()} seizure windows")

    # ── Chronological split ───────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = \
        chronological_split(X, y)

    print(f"\nSplit (chronological):")
    print(f"  Train: {X_train.shape}  seizures: {y_train.sum()}")
    print(f"  Val:   {X_val.shape}    seizures: {y_val.sum()}")
    print(f"  Test:  {X_test.shape}   seizures: {y_test.sum()}")

    # ── SMOTE on training set only ────────────────────────────────────────────
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # ── Save ANN dataset ──────────────────────────────────────────────────────
    np.savez_compressed(
        'data/processed/chb01_dataset_ann.npz',
        X_train=X_train_bal, y_train=y_train_bal,
        X_val=X_val,         y_val=y_val,
        X_test=X_test,       y_test=y_test
    )
    print("\nANN dataset saved: data/processed/chb01_dataset_ann.npz")

    # ── Same split for spike-encoded data (AKIDA) ─────────────────────────────
    spike_data = np.load('data/processed/chb01_spikes.npz')
    Xs, ys = spike_data['X'], spike_data['y']

    Xs_train, ys_train, Xs_val, ys_val, Xs_test, ys_test = \
        chronological_split(Xs, ys)
    Xs_train_bal, ys_train_bal = apply_smote(Xs_train, ys_train)

    np.savez_compressed(
        'data/processed/chb01_dataset_spikes.npz',
        X_train=Xs_train_bal, y_train=ys_train_bal,
        X_val=Xs_val,         y_val=ys_val,
        X_test=Xs_test,       y_test=ys_test
    )
    print("Spike dataset saved: data/processed/chb01_dataset_spikes.npz")

    # ── Summary stats to report in dissertation ───────────────────────────────
    print("\n=== Dataset summary (record in dissertation Table 1) ===")
    print(f"Patient          : chb01")
    print(f"Total windows    : {len(X)}")
    print(f"Window size      : 2.0 s @ 256 Hz = 512 samples")
    print(f"Channels         : {X.shape[1]} (18 common channels)")
    print(f"Overlap          : 50%")
    print(f"Split            : chronological 70/15/15")
    print(f"Train (balanced) : {len(X_train_bal)} "
          f"({y_train_bal.sum()} seizure / "
          f"{(y_train_bal==0).sum()} non-seizure)")
    print(f"Val (real dist.) : {len(X_val)} "
          f"({y_val.sum()} seizure)")
    print(f"Test (real dist.): {len(X_test)} "
          f"({y_test.sum()} seizure)")


