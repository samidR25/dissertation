"""
src/preprocessing/build_dataset_lopo_fold.py
==============================================
LOPO fold builder — pools ALL windows (full recording, not the 70% chronological
training slice) from every patient EXCEPT --leave-out into one training set.

Why full recording for pool members, not their chronological train split:
  In a rotating LOPO fold, a pool patient is NEVER evaluated in that fold — only
  the held-out patient is. There is therefore no leakage risk in using 100% of a
  pool member's data for training (the usual reason for a train/test split
  doesn't apply when nothing from that patient is ever scored). This also
  incidentally fixes two known data-availability quirks for free: chb11's zero
  train-split seizures (old 70% cut) and chb01's zero val/test seizures — both
  vanish once "pool member" no longer implies "individually chronologically
  split".

Undersample ratio default changed 10 -> 5 (supervisor-directed session,
9 July 2026): a bigger, 14-patient-per-fold pool supplies far more real seizure
windows in aggregate than the old 3-patient pool did, so less synthetic SMOTE
oversampling is needed to reach class balance for the same effective training
set. This is a disclosed, pre-registered choice — not tuned against results.
Comparing 10:1 vs 5:1 vs 3:1 on one fold is cheap and worth doing later if time
allows, to make the choice defensible rather than arbitrary.

Held-out patient itself is NEVER touched by this script — see
build_lopo_eval_set.py for its full-recording eval set.

Usage:
    python3 src/preprocessing/build_dataset_lopo_fold.py --leave-out chb10
    python3 src/preprocessing/build_dataset_lopo_fold.py --leave-out chb10 --us-ratio 5

Output:
    data/processed/multi_lopo_<leave_out>_dataset_ann.npz
        keys: X_train, y_train, domain_train, X_val, y_val, domain_val
        (same schema as multi_dataset_ann.npz -- train_baseline.py's existing
        --multi-patient fresh-training path reads this unchanged)
    data/processed/multi_lopo_<leave_out>_scaler.json
        {'patients': [...], 'domain_map': {...}, 'per_patient': {...}}
        (same schema as multi_scaler.json -- eval_event_level.py's Gate 1b
        already handles this format and correctly skips the consistency
        check for the held-out patient, since it's never a pool constituent)
"""
import argparse, json, os
import numpy as np

# Canonical 15-patient LOPO roster (everything currently preprocessed).
# Deliberately NOT curated down -- every patient here already has documented
# results in the ledger (including known negatives: chb03 structural failure,
# chb15 inert, chb16 genuine limitation). Dropping any of them now would look
# like cherry-picking regardless of intent, and conflicts with the project's
# own "negative results are documented, not omitted" rule.
DEFAULT_ROSTER = ['chb01', 'chb02', 'chb03', 'chb05', 'chb06', 'chb07',
                  'chb09', 'chb10', 'chb11', 'chb13', 'chb15', 'chb16',
                  'chb18', 'chb19', 'chb20']

parser = argparse.ArgumentParser()
parser.add_argument('--leave-out',  required=True,
                    help="Patient held out of the pool entirely this fold "
                         "(e.g. chb10). Must be in --patients.")
parser.add_argument('--patients',   nargs='+', default=DEFAULT_ROSTER,
                    help=f"Full roster BEFORE leave-out is removed. "
                         f"Default: {DEFAULT_ROSTER}")
parser.add_argument('--us-ratio',   type=int,   default=5,
                    help="Non-seizure to seizure ratio after undersampling "
                         "(default 5 -- see module docstring for rationale; "
                         "was 10 in the fixed 3-patient pool builder).")
parser.add_argument('--seed',       type=int,   default=42)
args = parser.parse_args()

if args.leave_out not in args.patients:
    raise ValueError(
        f"--leave-out {args.leave_out} is not in --patients {args.patients} "
        "-- nothing to hold out.")

pool_patients = [p for p in args.patients if p != args.leave_out]
rng = np.random.default_rng(args.seed)

print(f"LOPO fold        : held out = {args.leave_out}")
print(f"Pool patients ({len(pool_patients)}): {pool_patients}")
print(f"Undersample ratio: {args.us_ratio}:1  (non-seizure:seizure)")
print()

train_X_parts, train_y_parts = [], []
val_X_parts,   val_y_parts   = [], []
domain_train_parts, domain_val_parts = [], []
scaler_info = {}

for domain_id, pat in enumerate(pool_patients):
    X_path = f'data/processed/{pat}_X.npy'
    y_path = f'data/processed/{pat}_y.npy'
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"{X_path} not found.\n"
            f"Run: python3 src/preprocessing/preprocess.py --patient {pat}")

    X = np.load(X_path, mmap_mode='r')      # true memmap, no full-array RAM cost
    y = np.array(np.load(y_path), dtype=np.int32)
    N = len(y)
    n_seiz_full = int(y.sum())

    print(f"  {pat}: FULL recording, total={N} "
          f"(seizure={n_seiz_full}, {100*n_seiz_full/max(N,1):.3f}%)")

    if n_seiz_full == 0:
        raise ValueError(
            f"{pat} has zero seizure windows in its FULL recording -- "
            "cannot contribute to the pool. This would be a genuine data "
            "problem (not the old 70%-slice artifact), check preprocessing.")

    # ── Identify row indices over the FULL recording, undersample ─────────
    seiz_idx = np.where(y == 1)[0]
    nons_idx = np.where(y == 0)[0]
    n_keep   = min(len(nons_idx), n_seiz_full * args.us_ratio)
    kept_idx = rng.choice(nons_idx, size=n_keep, replace=False)
    sub_idx  = np.sort(np.concatenate([seiz_idx, kept_idx]))

    X_sub = np.array(X[sub_idx], dtype='float32')
    y_sub = y[sub_idx]
    print(f"          -> undersampled: {len(X_sub)} windows "
          f"(seizure={int(y_sub.sum())}, non-sz={int((y_sub==0).sum())})")

    # ── Scale to [0,255] using this patient's undersampled subset stats ───
    X_min = X_sub.min()
    X_max = X_sub.max()
    eps   = 1e-8
    scale = 255.0 / (X_max - X_min + eps)
    shift = -X_min * scale
    X_sub_s = (X_sub * scale + shift).clip(0, 255).astype('float32')
    scaler_info[pat] = {'scale': float(scale), 'shift': float(shift),
                        'X_min': float(X_min),  'X_max': float(X_max)}

    # ── Split THIS patient's undersampled subset 80/20 (real, pre-SMOTE) ──
    # Per-patient split (not a single pooled split) keeps every pool patient
    # represented in both train and val, same spirit as build_dataset_multi.py.
    from sklearn.model_selection import train_test_split
    n_seiz_sub = int(y_sub.sum())
    if n_seiz_sub < 2:
        raise ValueError(
            f"{pat}: only {n_seiz_sub} seizure window(s) after undersampling "
            "-- cannot stratified-split. Lower --us-ratio won't help here; "
            "check the raw seizure count for this patient.")
    X_tr_p, X_vl_p, y_tr_p, y_vl_p = train_test_split(
        X_sub_s, y_sub, test_size=0.20, stratify=y_sub, random_state=args.seed)

    train_X_parts.append(X_tr_p)
    train_y_parts.append(y_tr_p)
    val_X_parts.append(X_vl_p)
    val_y_parts.append(y_vl_p)
    domain_train_parts.append(np.full(len(y_tr_p), domain_id, dtype=np.int32))
    domain_val_parts.append(np.full(len(y_vl_p), domain_id, dtype=np.int32))

    del X, y, X_sub, X_sub_s, X_tr_p, X_vl_p

# ── Pool ─────────────────────────────────────────────────────────────────────
X_tr_real    = np.concatenate(train_X_parts, axis=0)
y_tr_real    = np.concatenate(train_y_parts, axis=0)
X_val        = np.concatenate(val_X_parts,   axis=0)
y_val        = np.concatenate(val_y_parts,   axis=0)
domain_tr_real = np.concatenate(domain_train_parts, axis=0)
domain_val     = np.concatenate(domain_val_parts,   axis=0)

print(f"\nPooled (before SMOTE): train={len(X_tr_real)} "
      f"(seizure={int(y_tr_real.sum())})  val={len(X_val)} "
      f"(seizure={int(y_val.sum())}, real, untouched by SMOTE)")

# ── SMOTE — applied ONLY to the real-train portion, PER DOMAIN ─────────────
# Same per-domain rationale as build_dataset_multi.py: a synthetic window
# interpolated between two real neighbours needs an unambiguous domain label,
# which per-domain SMOTE gives for free (every synthetic sample's domain is
# exactly the domain it was generated inside).
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
            f"Domain {dom_id} ({pool_patients[dom_id]}) has only "
            f"{n_seiz_dom} seizure window(s) in its real-train split -- "
            "cannot SMOTE within this domain.")
    X_flat_dom = X_dom.reshape(len(X_dom), -1)
    sm_dom = SMOTE(sampling_strategy='minority',
                   k_neighbors=min(5, n_seiz_dom - 1),
                   random_state=args.seed)
    X_sm_dom, y_sm_dom = sm_dom.fit_resample(X_flat_dom, y_dom)
    X_train_parts.append(X_sm_dom.reshape(-1, 18, 512).astype('float32'))
    y_train_parts.append(y_sm_dom.astype('int32'))
    domain_train_final_parts.append(np.full(len(y_sm_dom), dom_id, dtype=np.int32))
    print(f"  Domain {dom_id} ({pool_patients[dom_id]}): {len(y_dom)} real "
          f"-> {len(y_sm_dom)} post-SMOTE (seizure={int(y_sm_dom.sum())})")

X_train      = np.concatenate(X_train_parts, axis=0)
y_train      = np.concatenate(y_train_parts, axis=0)
domain_train = np.concatenate(domain_train_final_parts, axis=0)

print(f"\nAfter per-domain SMOTE: {len(X_train)} windows total "
      f"(seizure={int(y_train.sum())}, {100*y_train.mean():.1f}%)")

idx          = rng.permutation(len(X_train))
X_train      = X_train[idx]
y_train      = y_train[idx]
domain_train = domain_train[idx]

X_val = X_val.astype('float32')
y_val = y_val.astype('int32')

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
out_path = f'data/processed/multi_lopo_{args.leave_out}_dataset_ann.npz'
np.savez_compressed(out_path,
                    X_train=X_train, y_train=y_train, domain_train=domain_train,
                    X_val=X_val,     y_val=y_val,     domain_val=domain_val)
print(f"\nSaved  : {out_path}")
print(f"X_train: {X_train.shape}  range=[{X_train.min():.1f}, {X_train.max():.1f}]")
print(f"y_train: seizure={int(y_train.sum())} ({100*y_train.mean():.1f}%)")
print(f"X_val  : {X_val.shape}  seizure={int(y_val.sum())} ({100*y_val.mean():.1f}%)  (real)")

scaler_path = f'data/processed/multi_lopo_{args.leave_out}_scaler.json'
with open(scaler_path, 'w') as f:
    json.dump({'patients': pool_patients,
               'leave_out': args.leave_out,
               'us_ratio': args.us_ratio,
               'domain_map': {p: i for i, p in enumerate(pool_patients)},
               'per_patient': scaler_info}, f, indent=2)
print(f"Saved  : {scaler_path}")

assert X_train.shape[1:] == (18, 512)
assert 0 <= X_train.min() and X_train.max() <= 255
assert set(np.unique(y_train)) == {0, 1}
print("\n✓ Sanity checks passed")
print(f"-> Next: python3 src/models/train_baseline.py --model-version 2 "
      f"--multi-patient --pool-tag lopo_{args.leave_out}")
