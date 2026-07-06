"""
src/preprocessing/build_dataset_multi_longctx.py
==================================================
Gate 2 prerequisite: pools chb01+chb02+chb05's long-context windows into a
single base-training dataset, for fine-tuning to chb10 (Arms B/C). Mirrors
build_dataset_multi.py's memory-safe mmap pattern — each patient's raw
_X_longctx_w{ws}.npy is mmap-read (no full-array RAM cost), only the
undersampled train subset gets copied into RAM per patient.

Differs from build_dataset_multi.py in scaling: long-context channels have
wildly different native ranges (raw EEG ~1e-3, line-length ~1e-1,
delta/beta ratio ~10-40) — a single combined scale would crush the
smaller-magnitude channels. This reuses fit_scaler/apply_scaler from
build_dataset_stft.py (already generic per-channel over exactly 3
channels), fit ONCE on the pooled, post-SMOTE training set.

Does NOT save a chronological val_chrono reference array (unlike
build_dataset_multi.py, which does). Confirmed by direct inspection that
train_baseline.py never reads X_val_chrono/y_val_chrono from the npz at
all — it's diagnostic-only and unused downstream. Materializing and
concatenating it across 3 patients was also the actual cause of an OOM
("Killed") on an 11GB-RAM WSL2 box during this script's first real run:
the concatenate step needed source + destination arrays (~6.3GB each)
alive simultaneously for data nobody reads. Dropped entirely rather than
optimized, since optimizing something with zero downstream use is its
own kind of waste.

Run from ~/dissertation/, AFTER preprocess.py --longctx has built each
pool patient's raw windows at the target window size:
    python3 src/preprocessing/preprocess.py --patient chb01 --longctx --window-s 2.0 --longctx-lookback-s 12
    python3 src/preprocessing/preprocess.py --patient chb02 --longctx --window-s 2.0 --longctx-lookback-s 12
    python3 src/preprocessing/preprocess.py --patient chb05 --longctx --window-s 2.0 --longctx-lookback-s 12
    python3 src/preprocessing/build_dataset_multi_longctx.py --window-samples 512

Output:
    data/processed/multi_dataset_longctx_w{ws}.npz
        keys: X_train/y_train/X_val/y_val
    data/processed/multi_scaler_longctx_w{ws}.json
    data/processed/multi_dataset_longctx_w{ws}.npz.manifest.json
"""
import argparse
import json
import os
import sys

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

sys.path.insert(0, '.')
from src.preprocessing.build_dataset import chronological_split, undersample_train
from src.preprocessing.build_dataset_stft import fit_scaler, apply_scaler
from src.manifest import write_manifest, load_manifest


def smote_oversample_4d(X_sub: np.ndarray, y_sub: np.ndarray, random_state: int = 42):
    """4D-aware SMOTE — same logic as build_dataset_longctx.py's version."""
    n, c, t, f = X_sub.shape
    n_seiz = int(y_sub.sum())
    X_flat = X_sub.reshape(n, -1)
    sm = SMOTE(random_state=random_state, k_neighbors=min(5, n_seiz - 1))
    X_res, y_res = sm.fit_resample(X_flat, y_sub)
    return X_res.reshape(-1, c, t, f), y_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patients', nargs='+', default=['chb01', 'chb02', 'chb05'])
    parser.add_argument('--window-samples', type=int, required=True, choices=[512, 768])
    parser.add_argument('--train-frac', type=float, default=0.70)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--us-ratio', type=int, default=10,
                        help="Non-seizure to seizure ratio after undersampling")
    parser.add_argument('--seed', type=int, default=123,
                        help="Default 123 to match the original pooled "
                             "seed123 base's spirit — see Gate 2 chat notes; "
                             "this is a NEW base, not a literal continuation "
                             "of those weights, just seeded consistently.")
    parser.add_argument('--data-dir', default='data/processed/')
    args = parser.parse_args()

    ws = args.window_samples
    rng = np.random.default_rng(args.seed)
    print(f"Pooling patients : {args.patients}  (window_samples={ws})")
    print(f"Split            : train={args.train_frac:.0%} / val={args.val_frac:.0%}")
    print(f"Undersample ratio: {args.us_ratio}:1  (non-seizure:seizure)")
    print(f"Seed             : {args.seed}")
    print()

    train_real_parts, train_real_y_parts = [], []
    longctx_lookback_s = None
    window_s = None

    for pat in args.patients:
        X_path = os.path.join(args.data_dir, f'{pat}_X_longctx_w{ws}.npy')
        y_path = os.path.join(args.data_dir, f'{pat}_y_longctx_w{ws}.npy')
        if not (os.path.exists(X_path) and os.path.exists(y_path)):
            sys.exit(
                f"ERROR: {X_path} not found.\n"
                f"Run first: python3 src/preprocessing/preprocess.py "
                f"--patient {pat} --longctx --window-s <2.0 or 3.0> "
                f"--longctx-lookback-s 12"
            )

        # Provenance: confirm this patient's raw windows used the SAME
        # lookback as the others, and forward it — don't re-type it.
        pat_manifest = load_manifest(X_path, required=True)
        if longctx_lookback_s is None:
            longctx_lookback_s = pat_manifest['longctx_lookback_s']
            window_s = pat_manifest['window_s']
        elif pat_manifest['longctx_lookback_s'] != longctx_lookback_s:
            sys.exit(
                f"ERROR: {pat}'s manifest records longctx_lookback_s="
                f"{pat_manifest['longctx_lookback_s']}, but earlier "
                f"patient(s) used {longctx_lookback_s}. Refusing to pool "
                "windows built with inconsistent lookback."
            )

        X = np.load(X_path, mmap_mode='r')   # true memmap — plain .npy
        y = np.array(np.load(y_path), dtype=np.int32)
        N = len(y)

        # Chronological split (per patient) — train only matters here
        # (pool patients are never evaluated directly; chb10/13/15/16 etc.
        # are the held-out evaluation patients).
        n_train = int(N * args.train_frac)
        n_val = int(N * args.val_frac)
        y_tr = y[:n_train]
        y_vl = y[n_train:n_train + n_val]
        n_seiz = int(y_tr.sum())

        print(f"  {pat}: total={N}  train={n_train} "
              f"(seizure={n_seiz}, {100*n_seiz/max(n_train,1):.3f}%)  "
              f"val={n_val} (seizure={int(y_vl.sum())})")

        if n_seiz == 0:
            sys.exit(f"ERROR: {pat} training split has no seizure windows.")

        # ── Undersample (real, pre-SMOTE) — copies ONLY the selected rows ───
        seiz_idx = np.where(y_tr == 1)[0]
        nons_idx = np.where(y_tr == 0)[0]
        n_keep = min(len(nons_idx), n_seiz * args.us_ratio)
        kept_idx = rng.choice(nons_idx, size=n_keep, replace=False)
        sub_idx = np.sort(np.concatenate([seiz_idx, kept_idx]))

        X_sub = np.array(X[:n_train][sub_idx], dtype='float32')
        y_sub = y_tr[sub_idx]
        print(f"          -> undersampled: {len(X_sub)} windows "
              f"(seizure={int(y_sub.sum())}, non-sz={int((y_sub==0).sum())})")

        train_real_parts.append(X_sub)
        train_real_y_parts.append(y_sub)

        del X, y, X_sub

    # ── Pool (still raw/unscaled — scaling happens once, after SMOTE) ──────────
    X_pool_real = np.concatenate(train_real_parts, axis=0)
    y_pool_real = np.concatenate(train_real_y_parts, axis=0)
    del train_real_parts, train_real_y_parts

    n_seiz_pool = int(y_pool_real.sum())
    print(f"\nPooled (real, pre-SMOTE): {len(X_pool_real)} windows "
          f"(seizure={n_seiz_pool}, non-sz={int((y_pool_real==0).sum())})")

    # ── Split BEFORE SMOTE (stratified-random — same leak-avoidance pattern
    #    as build_dataset_multi.py) ──────────────────────────────────────────────
    X_tr_real, X_vl_real, y_tr_real, y_vl_real = train_test_split(
        X_pool_real, y_pool_real, test_size=0.20,
        stratify=y_pool_real, random_state=args.seed,
    )
    print(f"Real (pre-SMOTE) split: train={len(X_tr_real)} "
          f"(seizure={int(y_tr_real.sum())})  val={len(X_vl_real)} "
          f"(seizure={int(y_vl_real.sum())})")

    # ── SMOTE — train portion only ──────────────────────────────────────────────
    X_train_bal, y_train_bal = smote_oversample_4d(X_tr_real, y_tr_real, args.seed)
    print(f"After SMOTE: {len(X_train_bal)} windows "
          f"(seizure={int(y_train_bal.sum())}, {100*y_train_bal.mean():.1f}%)")

    perm = rng.permutation(len(X_train_bal))
    X_train_bal = X_train_bal[perm]
    y_train_bal = y_train_bal[perm]

    # ── Fit ONE per-channel scaler on the pooled, balanced training set ────────
    print("\nFitting per-channel scaler on pooled (balanced) train set...")
    scaler = fit_scaler(X_train_bal)
    ch_names = {0: 'raw EEG', 1: 'rolling line-length', 2: 'rolling delta/beta ratio'}
    for ch in range(3):
        print(f"  ch{ch} {ch_names[ch]}: "
              f"[{scaler[f'ch{ch}_min']:.4f}, {scaler[f'ch{ch}_max']:.4f}] -> [0,255]")

    X_train_sc = apply_scaler(X_train_bal, scaler)
    X_val_sc = apply_scaler(X_vl_real, scaler)

    assert X_train_sc.min() >= 0.0 and X_train_sc.max() <= 255.01
    print(f"  Scaled range check: [{X_train_sc.min():.1f}, {X_train_sc.max():.1f}]  OK")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.data_dir, exist_ok=True)
    out_path = os.path.join(args.data_dir, f'multi_dataset_longctx_w{ws}.npz')
    scaler_path = os.path.join(args.data_dir, f'multi_scaler_longctx_w{ws}.json')

    print(f"\nSaving {out_path} ...")
    np.savez_compressed(
        out_path,
        X_train=X_train_sc, y_train=y_train_bal,
        X_val=X_val_sc, y_val=y_vl_real.astype('int32'),
    )
    with open(scaler_path, 'w') as f:
        json.dump(scaler, f, indent=2)

    write_manifest(
        out_path,
        patients=args.patients,
        window_s=window_s,
        window_samples=ws,
        longctx_lookback_s=longctx_lookback_s,
        scaler_path=scaler_path,
        scaler=scaler,
        seed=args.seed,
        us_ratio=args.us_ratio,
    )

    print(f"\n{'='*55}")
    print(f"DONE — pooled longctx dataset (window_samples={ws})")
    print(f"  {out_path}")
    print(f"  Scaler  : {scaler_path}")
    print(f"  X_train : {X_train_sc.shape}   seizure={int(y_train_bal.sum())} "
          f"({100*y_train_bal.mean():.1f}%)")
    print(f"  X_val   : {X_val_sc.shape}    seizure={int(y_vl_real.sum())}  (real, pre-SMOTE)")
    print(f"  Shape check: {X_train_sc.shape[1:]}  (expect (18, {ws}, 3))")
    print(f"{'='*55}")
    print(f"\nNext: python3 src/models/train_baseline.py --multi-patient "
          f"--longctx --window-samples {ws} --seed {args.seed}")


if __name__ == '__main__':
    main()
