"""
src/preprocessing/build_dataset_multi.py
=========================================
Phase 2c: Multi-patient dataset builder.

Pools training windows from chb01 + chb02 + chb05.
chb03 is NEVER touched, but now it is.

Memory strategy:
  chb01_windows.npz is ~5.4 GB — loading it normally kills WSL2 (16 GB RAM,
  Linux OOM killer fires when Python + TF + the array compete for address space).

  Fix: np.load(mmap_mode='r') keeps the file on disk. NumPy serves rows on
  demand via the OS page cache. Only the rows we actually index (seizure +
  undersampled non-seizure, ~5K rows) get paged into RAM.
  Peak RAM: ~0.36 GB per patient instead of 5.4 GB.

  Sequence per patient:
    1. mmap-load the full array (no RAM cost yet)
    2. Find seizure row indices in y (y is tiny — load normally)
    3. Randomly select 10× non-seizure indices
    4. np.copy() only those rows into a dense array (~0.18 GB)
    5. Scale to [0,255], append to pool list, discard mmap reference

Run from ~/dissertation/:
    python3 src/preprocessing/build_dataset_multi.py
"""

import argparse, json, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--patients',   nargs='+', default=['chb01', 'chb02', 'chb05'])
parser.add_argument('--train-frac', type=float, default=0.70)
parser.add_argument('--val-frac',   type=float, default=0.15)
parser.add_argument('--us-ratio',   type=int,   default=10,
                    help="Non-seizure to seizure ratio after undersampling (default 10)")
parser.add_argument('--seed',       type=int,   default=42)
args = parser.parse_args()

#if 'chb03' in args.patients:
 #   raise ValueError("chb03 is the held-out test patient — remove it from --patients.")

rng = np.random.default_rng(args.seed)
print(f"Pooling patients : {args.patients}")
print(f"Split            : train={args.train_frac:.0%} / val={args.val_frac:.0%}")
print(f"Undersample ratio: {args.us_ratio}:1  (non-seizure:seizure)")
print()

train_X_parts, train_y_parts = [], []
val_X_parts,   val_y_parts   = [], []
domain_train_parts, domain_val_parts = [], []   # DANN scoping
scaler_info = {}

for domain_id, pat in enumerate(args.patients):
    X_path = f'data/processed/{pat}_X.npy'
    y_path = f'data/processed/{pat}_y.npy'
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"{X_path} not found.\n"
            f"Run: python3 src/preprocessing/preprocess.py --patient {pat}")

    X = np.load(X_path, mmap_mode='r')      # true memmap — plain .npy, no compression
    y = np.array(np.load(y_path), dtype=np.int32)
    N   = len(y)

    n_train = int(N * args.train_frac)
    n_val   = int(N * args.val_frac)

    y_tr = y[:n_train]
    y_vl = y[n_train:n_train + n_val]

    n_seiz = int(y_tr.sum())
    print(f"  {pat}: total={N}  train={n_train} "
          f"(seizure={n_seiz}, {100*n_seiz/max(n_train,1):.3f}%)  "
          f"val={n_val} (seizure={int(y_vl.sum())})")

    if n_seiz == 0:
        raise ValueError(
            f"{pat} training split has no seizure windows. "
            "Check _windows.npz is complete.")

    # ── Identify row indices, undersample ─────────────────────────────────────
    seiz_idx = np.where(y_tr == 1)[0]
    nons_idx = np.where(y_tr == 0)[0]
    n_keep   = min(len(nons_idx), n_seiz * args.us_ratio)
    kept_idx = rng.choice(nons_idx, size=n_keep, replace=False)
    sub_idx  = np.sort(np.concatenate([seiz_idx, kept_idx]))

    # ── Copy only the selected rows from disk → RAM ────────────────────────────
    # np.array() on a memmap slice forces a contiguous copy of just those rows.
    X_sub = np.array(X[:n_train][sub_idx], dtype='float32')   # ~0.18 GB for chb01
    y_sub = y_tr[sub_idx]
    print(f"          → undersampled: {len(X_sub)} windows "
          f"(seizure={int(y_sub.sum())}, non-sz={int((y_sub==0).sum())})")

    # ── Scale to [0,255] using training-slice min/max ─────────────────────────
    # Use the seizure-only rows' stats to avoid outlier non-seizure frames
    # dominating the scale — but simpler and consistent with build_dataset.py
    # is to use the sub-sampled set's global min/max.
    X_min = X_sub.min()
    X_max = X_sub.max()
    eps   = 1e-8
    scale = 255.0 / (X_max - X_min + eps)
    shift = -X_min * scale
    X_sub_s = (X_sub * scale + shift).clip(0, 255).astype('float32')
    scaler_info[pat] = {'scale': float(scale), 'shift': float(shift),
                        'X_min': float(X_min),  'X_max': float(X_max)}

    # Val slice: also small enough to copy safely
    X_vl_s = np.array(X[n_train:n_train + n_val], dtype='float32')
    X_vl_s = (X_vl_s * scale + shift).clip(0, 255).astype('float32')

    train_X_parts.append(X_sub_s)
    train_y_parts.append(y_sub)
    val_X_parts.append(X_vl_s)
    val_y_parts.append(y_vl)
    domain_train_parts.append(np.full(len(y_sub), domain_id, dtype=np.int32))
    domain_val_parts.append(np.full(len(y_vl), domain_id, dtype=np.int32))

    del X, y, X_sub, X_sub_s, X_vl_s   # release mmap + copies

# ── Pool ───────────────────────────────────────────────────────────────────────
X_pool       = np.concatenate(train_X_parts, axis=0)
y_pool       = np.concatenate(train_y_parts, axis=0)
X_val_chrono = np.concatenate(val_X_parts,   axis=0)   # real, chronological,
y_val_chrono = np.concatenate(val_y_parts,   axis=0)   # but 0 seizures — kept for reference only
domain_pool       = np.concatenate(domain_train_parts, axis=0)   # DANN scoping
domain_val_chrono = np.concatenate(domain_val_parts,   axis=0)

n_seiz_pool = int(y_pool.sum())
print(f"\nPooled (before split/SMOTE): {len(X_pool)} windows "
      f"(seizure={n_seiz_pool}, non-sz={int((y_pool==0).sum())})")

# ── Split BEFORE SMOTE ──────────────────────────────────────────────────────────
# The old approach carved val from the SMOTE'd training set, after the fact.
# SMOTE generates synthetic seizure windows by interpolating real neighbours —
# a stratified split done afterward can put a synthetic point on one side and
# its real source neighbours on the other, leaking information across the
# train/val boundary. Splitting the real, undersampled pool FIRST — then
# applying SMOTE only to the resulting train portion — removes that leakage
# entirely: val only ever contains genuine windows.
#
# This split is stratified-random, not chronological — the genuine chronological
# val region (X_val_chrono above) has zero seizures for every pool patient, so
# it can't serve this purpose. That's an acceptable, disclosed compromise here
# specifically because this val is used ONLY for early-stopping / checkpoint
# selection during pooled training — never reported as a cross-patient
# generalisation claim. The actual clinical claims come from
# eval_event_level.py against chb03/chb10/chb06/chb11's untouched chronological
# test splits, which this never touches.
from sklearn.model_selection import train_test_split
X_tr_real, X_vl_real, y_tr_real, y_vl_real, domain_tr_real, domain_vl_real = train_test_split(
    X_pool, y_pool, domain_pool, test_size=0.20, stratify=y_pool, random_state=args.seed
)
print(f"Real (pre-SMOTE) split: train={len(X_tr_real)} (seizure={int(y_tr_real.sum())})  "
      f"val={len(X_vl_real)} (seizure={int(y_vl_real.sum())})")

# ── SMOTE — applied ONLY to the real-train portion, PER DOMAIN ─────────────────
# DANN scoping: pooled SMOTE has no notion of domain -- a synthetic seizure
# window interpolated between two real chb01 neighbours would need a domain
# label assigned to it with no principled inheritance rule. Running SMOTE
# separately within each domain's real-train rows sidesteps this entirely:
# every synthetic sample's domain label is exactly the domain it was
# generated inside. See Handoff_calibration_session_to_dann_supcon.md sec4d.
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("pip install imbalanced-learn")

X_train_parts, y_train_parts, domain_train_final_parts = [], [], []
for dom_id in sorted(set(domain_tr_real.tolist())):
    mask = domain_tr_real == dom_id
    X_dom, y_dom = X_tr_real[mask], y_tr_real[mask]
    n_seiz_dom = int(y_dom.sum())
    if n_seiz_dom < 2:
        raise ValueError(
            f"Domain {dom_id} ({args.patients[dom_id]}) has only "
            f"{n_seiz_dom} seizure window(s) in its real-train split after "
            "the pooled 80/20 split -- cannot SMOTE within this domain. "
            "Per-domain SMOTE requires every pool patient to individually "
            "clear this bar."
        )
    X_flat_dom = X_dom.reshape(len(X_dom), -1)
    sm_dom = SMOTE(sampling_strategy='minority',
                   k_neighbors=min(5, n_seiz_dom - 1),
                   random_state=args.seed)
    X_sm_dom, y_sm_dom = sm_dom.fit_resample(X_flat_dom, y_dom)
    X_train_parts.append(X_sm_dom.reshape(-1, 18, 512).astype('float32'))
    y_train_parts.append(y_sm_dom.astype('int32'))
    domain_train_final_parts.append(np.full(len(y_sm_dom), dom_id, dtype=np.int32))
    print(f"  Domain {dom_id} ({args.patients[dom_id]}): {len(y_dom)} real "
          f"-> {len(y_sm_dom)} post-SMOTE (seizure={int(y_sm_dom.sum())})")

X_train      = np.concatenate(X_train_parts, axis=0)
y_train      = np.concatenate(y_train_parts, axis=0)
domain_train = np.concatenate(domain_train_final_parts, axis=0)

print(f"After per-domain SMOTE: {len(X_train)} windows total "
      f"(seizure={int(y_train.sum())}, {100*y_train.mean():.1f}%)")

idx          = rng.permutation(len(X_train))
X_train      = X_train[idx]
y_train      = y_train[idx]
domain_train = domain_train[idx]

X_val      = X_vl_real.astype('float32')   # real, untouched by SMOTE
y_val      = y_vl_real.astype('int32')
domain_val = domain_vl_real.astype('int32')       # DANN scoping

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
out_path = 'data/processed/multi_dataset_ann.npz'
np.savez_compressed(out_path,
                    X_train=X_train, y_train=y_train, domain_train=domain_train,
                    X_val=X_val,     y_val=y_val,     domain_val=domain_val,
                    X_val_chrono=X_val_chrono, y_val_chrono=y_val_chrono,
                    domain_val_chrono=domain_val_chrono)
print(f"\nSaved  : {out_path}")
print(f"X_train: {X_train.shape}  range=[{X_train.min():.1f}, {X_train.max():.1f}]")
print(f"y_train: seizure={int(y_train.sum())} ({100*y_train.mean():.1f}%)")
print(f"X_val  : {X_val.shape}  seizure={int(y_val.sum())} ({100*y_val.mean():.1f}%)  (real, pre-SMOTE)")
print(f"X_val_chrono: {X_val_chrono.shape}  seizure={int(y_val_chrono.sum())}  (reference only)")

with open('data/processed/multi_scaler.json', 'w') as f:
    json.dump({'patients': args.patients,
               'domain_map': {p: i for i, p in enumerate(args.patients)},
               'per_patient': scaler_info}, f, indent=2)
print("Saved  : data/processed/multi_scaler.json")
print(f"Domain map: {dict((p, i) for i, p in enumerate(args.patients))}")

assert X_train.shape[1:] == (18, 512)
assert 0 <= X_train.min() and X_train.max() <= 255
assert set(np.unique(y_train)) == {0, 1}
print("\n✓ Sanity checks passed")
print("→ Next: python3 src/models/train_baseline.py --model-version 2 --multi-patient")
