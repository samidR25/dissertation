"""
src/preprocessing/build_dataset_longctx.py
============================================
Gate 1 — split/balance/scale pipeline for the long-context 3-channel
dataset (chb10 only, both window sizes —
Handoff_architecture_scoping_to_implementation.md §4).

Deliberately does NOT mirror build_dataset_stft.py's "add channels on top
of an already-split/already-windowed dataset" pattern — the handoff is
explicit that pattern doesn't work for long-lookback channels (the
lookback must reach back before each window's start, which no longer
exists post-windowing). Instead this mirrors build_dataset.py's
split/undersample/SMOTE/scale pipeline, applied to the long-context 4D
windows that preprocess.py --longctx already built and windowed correctly.

Reuses (does not duplicate):
  chronological_split, undersample_train  — from build_dataset.py
  fit_scaler, apply_scaler                — from build_dataset_stft.py
                                             (already generic over 3 channels)
SMOTE is reimplemented locally only because build_dataset.py's
smote_oversample() assumes a 3D (n, ch, t) input; this is 4D (n, 18, t, 3).

Saves X_train_real/y_train_real (the real, undersampled, PRE-SMOTE train
pool) alongside the usual keys — required by train_baseline.py's
--freeze-depth fine-tune path (_stratified_real_split, Gate 2d
convention). Earlier versions of this script omitted these keys.

Input  (from preprocess.py --longctx):
  data/processed/{patient}_X_longctx_w{window_samples}.npy   (N,18,ws,3) f32
  data/processed/{patient}_y_longctx_w{window_samples}.npy   (N,) int32
  data/processed/{patient}_X_longctx_w{window_samples}.npy.manifest.json

Output:
  data/processed/{patient}_dataset_longctx_w{window_samples}.npz
      keys: X_train/y_train/X_train_real/y_train_real/X_val/y_val/X_test/y_test
  data/processed/{patient}_scaler_longctx_w{window_samples}.json
  data/processed/{patient}_dataset_longctx_w{window_samples}.npz.manifest.json
      records window_samples + longctx_lookback_s forwarded from the
      preprocessing manifest above — provenance chain, not re-typed.

Usage:
  python3 src/preprocessing/build_dataset_longctx.py --patient chb10 --window-samples 512
  python3 src/preprocessing/build_dataset_longctx.py --patient chb10 --window-samples 768
"""
import argparse
import json
import os
import sys

import numpy as np
from imblearn.over_sampling import SMOTE

sys.path.insert(0, '.')
from src.preprocessing.build_dataset import chronological_split, undersample_train
from src.preprocessing.build_dataset_stft import fit_scaler, apply_scaler
from src.manifest import write_manifest, load_manifest


def smote_oversample_4d(X_sub: np.ndarray, y_sub: np.ndarray,
                        random_state: int = 42):
    """
    4D-aware SMOTE. Same logic as build_dataset.py's smote_oversample(),
    generalised to (n, 18, window_samples, 3) instead of (n, ch, t).
    """
    n, c, t, f = X_sub.shape
    n_seiz = int(y_sub.sum())
    X_flat = X_sub.reshape(n, -1)
    sm = SMOTE(random_state=random_state, k_neighbors=min(5, n_seiz - 1))
    X_res, y_res = sm.fit_resample(X_flat, y_sub)
    X_res = X_res.reshape(-1, c, t, f)
    print(f"  After SMOTE:  {y_res.sum()} seizure, "
          f"{(y_res==0).sum()} non-seizure")
    return X_res, y_res


def build_longctx_dataset(patient: str, window_samples: int,
                          data_dir: str = 'data/processed/',
                          train_frac: float = 0.70, val_frac: float = 0.15):
    X_path = os.path.join(data_dir, f'{patient}_X_longctx_w{window_samples}.npy')
    y_path = os.path.join(data_dir, f'{patient}_y_longctx_w{window_samples}.npy')
    assert os.path.exists(X_path) and os.path.exists(y_path), (
        f"Long-context windows not found: {X_path}\n"
        f"Run first: python3 src/preprocessing/preprocess.py --patient {patient} "
        f"--longctx --window-s <...> --longctx-lookback-s <...>"
    )

    # ── Provenance: read forward, don't re-type ───────────────────────────────
    src_manifest = load_manifest(X_path, required=True)
    longctx_lookback_s = src_manifest['longctx_lookback_s']
    window_s = src_manifest['window_s']
    assert src_manifest['window_samples'] == window_samples, (
        f"ERROR: manifest at {X_path}.manifest.json records "
        f"window_samples={src_manifest['window_samples']}, but this script "
        f"was called with --window-samples {window_samples}. Refusing to "
        f"proceed on a mismatch (chb10ft scaler-class bug pattern)."
    )

    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path)
    print(f"\nPatient: {patient}  (longctx, window_samples={window_samples}, "
          f"window_s={window_s}, lookback_s={longctx_lookback_s})")
    print(f"Loaded: {X.shape}  |  {y.sum()} seizure windows "
          f"({100*y.mean():.2f}%)")

    # ── Chronological split ───────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = \
        chronological_split(X, y, train_frac, val_frac)
    print(f"\nSplit (chronological {int(train_frac*100)}/{int(val_frac*100)}/"
          f"{int((1-train_frac-val_frac)*100)}):")
    print(f"  Train : {X_train.shape}  seizures: {y_train.sum()}")
    print(f"  Val   : {X_val.shape}    seizures: {y_val.sum()}")
    print(f"  Test  : {X_test.shape}   seizures: {y_test.sum()}")

    # ── Undersample + SMOTE on training set only ──────────────────────────────
    if y_train.sum() == 0:
        print(f"\nWARNING: {patient} training split has ZERO seizure windows. "
              "Skipping SMOTE — saved npz is still valid for held-out eval.")
        X_train_real, y_train_real = X_train, y_train
        X_train_bal, y_train_bal = X_train, y_train
    else:
        print("\nApplying undersample + SMOTE to training set...")
        X_train_real, y_train_real = undersample_train(X_train, y_train)
        X_train_bal, y_train_bal = smote_oversample_4d(X_train_real, y_train_real)

    # ── Per-channel [0,255] scaling (reuses build_dataset_stft.py's fit/apply,
    #    already generic over exactly 3 channels) ─────────────────────────────
    print("\nFitting per-channel scaler on (balanced) train split...")
    scaler = fit_scaler(X_train_bal)
    ch_names = {0: 'raw EEG', 1: 'rolling line-length', 2: 'rolling delta/beta ratio'}
    for ch in range(3):
        print(f"  ch{ch} {ch_names[ch]}: "
              f"[{scaler[f'ch{ch}_min']:.3f}, {scaler[f'ch{ch}_max']:.3f}] -> [0,255]")

    X_train_sc = apply_scaler(X_train_bal, scaler)
    X_train_real_sc = apply_scaler(X_train_real, scaler)
    X_val_sc = apply_scaler(X_val, scaler)
    X_test_sc = apply_scaler(X_test, scaler)

    assert X_train_sc.min() >= 0.0 and X_train_sc.max() <= 255.01, \
        "Scale error on train"
    print(f"  Scaled range check: [{X_train_sc.min():.1f}, {X_train_sc.max():.1f}]  OK")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(data_dir, f'{patient}_dataset_longctx_w{window_samples}.npz')
    scaler_path = os.path.join(data_dir, f'{patient}_scaler_longctx_w{window_samples}.json')

    print(f"\nSaving {out_path} ...")
    np.savez_compressed(
        out_path,
        X_train=X_train_sc, y_train=y_train_bal,
        X_train_real=X_train_real_sc, y_train_real=y_train_real,
        X_val=X_val_sc, y_val=y_val,
        X_test=X_test_sc, y_test=y_test,
    )
    with open(scaler_path, 'w') as f:
        json.dump(scaler, f, indent=2)

    write_manifest(
        out_path,
        patient=patient,
        window_s=window_s,
        window_samples=window_samples,
        longctx_lookback_s=longctx_lookback_s,
        scaler_path=scaler_path,
        scaler=scaler,
        split_ratios=[train_frac, val_frac, round(1 - train_frac - val_frac, 4)],
        split_counts={'train': int(len(X_train)), 'val': int(len(X_val)),
                     'test': int(len(X_test))},
        smote_applied=bool(y_train.sum() != 0),
        smote_undersample_ratio=10,
        source_manifest=f'{X_path}.manifest.json',
    )

    print(f"\n{'='*55}")
    print(f"DONE — {patient} long-context dataset (window_samples={window_samples})")
    print(f"  {out_path}")
    print(f"  Manifest: {out_path}.manifest.json")
    print(f"  X_train: {X_train_sc.shape}   [{X_train_sc.min():.0f},{X_train_sc.max():.0f}]")
    print(f"  X_train_real: {X_train_real_sc.shape}  (real, pre-SMOTE)")
    print(f"  X_val:   {X_val_sc.shape}    seizures={y_val.sum()}")
    print(f"  X_test:  {X_test_sc.shape}   seizures={y_test.sum()}")
    print(f"  Shape check: {X_train_sc.shape[1:]}  (expect (18, {window_samples}, 3))")
    print(f"{'='*55}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', required=True)
    parser.add_argument('--window-samples', type=int, required=True,
                        choices=[512, 768])
    parser.add_argument('--data-dir', default='data/processed/')
    args = parser.parse_args()

    build_longctx_dataset(args.patient, args.window_samples, args.data_dir)
