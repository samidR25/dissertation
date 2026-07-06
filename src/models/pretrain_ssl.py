"""
src/models/pretrain_ssl.py
============================
Candidate C-i first experiment (Handoff_post_dann_scoping_to_implementation.md
sec5): masked-window reconstruction self-supervised pretraining, targeting
RC2 (small labeled pool) by using the pool's ABUNDANT unlabeled/non-seizure
data differently, not more patients. Genuinely different lever than pool
expansion (already confirmed negative) or DANN/CORAL (both target RC1, not
RC2) -- this uses the EXISTING 3-patient pool's non-seizure windows to give
the trunk a representation not dominated by SMOTE's synthetic minority-class
geometry, before the supervised head is ever attached.

SCOPE BOUNDARY (sec5b, explicit, NOT renegotiable without Dr. Pham sign-off):
pretraining uses ONLY chb01/02/05's own non-seizure windows -- the SAME
three pool patients C1/A/CORAL already use. Zero held-out patient data
touched. This preserves the current C1 definition ("zero patient data").
A more aggressive transductive version (pretrain on held-out patients'
unlabeled EEG too) is explicitly NOT implemented here -- that changes the
C1 definition and needs a framing conversation with Dr. Pham first, same
standard as CORAL's A-ii.

SIMPLIFICATION vs a full masked-autoencoder (disclosed, not hidden): masks
a FIXED time-span (the middle --mask-frac of every window, same relative
position for every sample) rather than a random position per sample/epoch.
Avoids needing to communicate mask position to the decoder (no positional
encoding) at the cost of a less rigorous pretext task than a true MAE.
Reasonable for a cheap first screen (sec5e) -- revisit (random position,
per-epoch remasking) only if C-i shows genuine signal.

CONSISTENCY REQUIREMENT: pretraining scales inputs under the SAME
per-patient scaler the supervised pool already uses (multi_scaler.json's
per_patient entries), NOT a freshly-computed one. If the pretrained
trunk saw a different input distribution than the downstream supervised
fine-tune, the whole point of pretraining (giving the trunk a head start
on THIS pool's actual feature scale) would be silently undermined. This
script hard-refuses if multi_scaler.json doesn't exist yet -- run
build_dataset_multi.py first if needed (it will exist already for anyone
who ran CORAL/DANN).

Run from ~/dissertation/ with akida_env activated:
    python3 src/models/pretrain_ssl.py --seed 123 --mask-frac 0.25

Then feed the result into train_baseline.py's new --init-from-ssl flag
(applied by apply_ssl_pretrain_init_patch.py):
    python3 src/models/train_baseline.py --model-version 2 --multi-patient \\
        --init-from-ssl results/pretrained_trunk_ssl_v2.h5 --seed 123
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import argparse, json, os, sys
import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.model_selection import train_test_split

sys.path.insert(0, '.')

parser = argparse.ArgumentParser()
parser.add_argument('--patients', nargs='+', default=['chb01', 'chb02', 'chb05'],
                    help="Pool patients (sec5b: must stay chb01/02/05 -- "
                         "no held-out patient data without a separate "
                         "Dr. Pham sign-off conversation)")
parser.add_argument('--n-per-patient', type=int, default=6000,
                    help="Max non-seizure windows sampled per patient "
                         "(memory guard, same mmap-then-subsample pattern "
                         "as build_dataset_multi.py -- default keeps total "
                         "pool ~650MB, safe on WSL2's 16GB)")
parser.add_argument('--mask-frac', type=float, default=0.25,
                    help="Fraction of the 512-sample window masked, as one "
                         "FIXED contiguous span centered in the window "
                         "(same position for every sample -- see module "
                         "docstring on why this is simplified vs a true MAE)")
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

if set(args.patients) != {'chb01', 'chb02', 'chb05'}:
    print(f"\nWARNING: --patients {args.patients} differs from the sec5b "
          "default scope (chb01/02/05). If any of these are held-out "
          "evaluation patients, STOP -- that changes the C1 definition "
          "and needs a Dr. Pham sign-off conversation first (same standard "
          "as CORAL's A-ii). Proceeding on the assumption you know this.")

random_seed = args.seed
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
rng = np.random.default_rng(random_seed)
print(f"Random seed: {random_seed}")

WINDOW_SAMPLES = 512
N_CHANNELS = 18
mask_len   = int(WINDOW_SAMPLES * args.mask_frac)
mask_start = (WINDOW_SAMPLES - mask_len) // 2
mask_end   = mask_start + mask_len
print(f"Mask span: samples [{mask_start}:{mask_end}] "
      f"({mask_len}/{WINDOW_SAMPLES} = {100*args.mask_frac:.0f}% of window, "
      "fixed position for every sample)")

# ── Load the SAME per-patient scaler the supervised pool uses ────────────────
scaler_path = 'data/processed/multi_scaler.json'
if not os.path.exists(scaler_path):
    sys.exit(
        f"ERROR: {scaler_path} not found -- run "
        "build_dataset_multi.py --patients chb01 chb02 chb05 first (needed "
        "so pretraining scales inputs identically to the downstream "
        "supervised training, see module docstring)."
    )
with open(scaler_path) as f:
    multi_scaler = json.load(f)
if 'per_patient' not in multi_scaler:
    sys.exit(f"ERROR: {scaler_path} has no 'per_patient' scaler map -- "
             "regenerate via build_dataset_multi.py.")

# ── Load non-seizure windows only, per patient, mmap-then-subsample ──────────
# Same memory strategy as build_dataset_multi.py's own docstring: mmap the
# full array (no RAM cost), find indices, copy only the subsampled rows.
pool_parts = []
for pat in args.patients:
    X_path = f'data/processed/{pat}_X.npy'
    y_path = f'data/processed/{pat}_y.npy'
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        sys.exit(f"ERROR: {X_path} not found.\n"
                 f"Run: python3 src/preprocessing/preprocess.py --patient {pat}")
    if pat not in multi_scaler['per_patient']:
        sys.exit(f"ERROR: {pat} has no entry in {scaler_path}'s per_patient "
                 "map -- scaler provenance mismatch, refusing to guess.")
    scale = multi_scaler['per_patient'][pat]['scale']
    shift = multi_scaler['per_patient'][pat]['shift']

    X = np.load(X_path, mmap_mode='r')
    y = np.array(np.load(y_path), dtype=np.int32)
    nonseiz_idx = np.where(y == 0)[0]
    n_keep = min(len(nonseiz_idx), args.n_per_patient)
    kept_idx = rng.choice(nonseiz_idx, size=n_keep, replace=False)
    kept_idx = np.sort(kept_idx)

    X_sub = np.array(X[kept_idx], dtype='float32')
    X_sub_scaled = (X_sub * scale + shift).clip(0, 255).astype('float32')
    pool_parts.append(X_sub_scaled)
    print(f"  {pat}: {len(nonseiz_idx)} non-seizure windows available, "
          f"kept {n_keep} (scale={scale:.4f}, shift={shift:.2f}, "
          f"matches {scaler_path})")
    del X, y, X_sub

X_pool = np.concatenate(pool_parts, axis=0)
rng.shuffle(X_pool)
print(f"\nPooled non-seizure windows: {len(X_pool)}  shape={X_pool.shape}")
assert X_pool.shape[1:] == (N_CHANNELS, WINDOW_SAMPLES), \
    f"Unexpected window shape {X_pool.shape[1:]}"

# ── Build masked input / reconstruction target ────────────────────────────────
# Mask -> 0.0 (the input range's own minimum after scale_255 clipping, so
# "masked" isn't an out-of-distribution value the network has never seen).
# Target normalised to [0,1] (divide by 255) for numerically well-behaved
# MSE gradients -- the decoder's linear output head isn't range-constrained,
# it just learns to land near this normalised target.
target = X_pool[:, :, mask_start:mask_end].copy() / 255.0   # (N, 18, mask_len)
X_masked = X_pool.copy()
X_masked[:, :, mask_start:mask_end] = 0.0
X_masked = X_masked[..., np.newaxis].astype('float32')       # (N, 18, 512, 1)
target = target.astype('float32')

X_tr, X_vl, y_tr, y_vl = train_test_split(
    X_masked, target, test_size=0.10, random_state=random_seed)
print(f"Train: {len(X_tr)}   Val (early-stopping only, not a reported "
      f"metric): {len(X_vl)}")

# Trivial baseline: predicting the calibration set's own global mean patch
# for every sample -- the "did the pretext task learn anything" floor,
# same discipline as CORAL's covariance-distance readout / DANN's domain-
# accuracy control (prove it moved something, don't assume it).
baseline_mse = float(np.mean((y_vl - y_tr.mean(axis=0, keepdims=True)) ** 2))
print(f"\n[SSL] Trivial baseline (predict train-mean patch for every val "
      f"sample): MSE = {baseline_mse:.6f}")

# ── Build pretext-task model (trunk identical to build_seizure_cnn_v2) ───────
from src.models.akida_cnn_v2 import build_seizure_cnn_v2_ssl_pretrain, extract_pretrained_trunk

ssl_model = build_seizure_cnn_v2_ssl_pretrain(
    n_channels=N_CHANNELS, window_samples=WINDOW_SAMPLES, mask_len=mask_len)
ssl_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
ssl_model.summary(print_fn=lambda x: print(f"  {x}"))

os.makedirs('results', exist_ok=True)
ssl_ckpt = f'results/ssl_pretrain_TRAINING_v2.h5'
callbacks = [
    keras.callbacks.ModelCheckpoint(
        ssl_ckpt, monitor='val_loss', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]
ssl_model.fit(
    X_tr, y_tr, validation_data=(X_vl, y_vl),
    epochs=args.epochs, batch_size=args.batch, callbacks=callbacks, verbose=2,
)

best_ssl = keras.models.load_model(ssl_ckpt, compile=False)
val_pred = best_ssl.predict(X_vl, verbose=0, batch_size=128)
final_val_mse = float(np.mean((val_pred - y_vl) ** 2))
print(f"\n[SSL] Final val reconstruction MSE: {final_val_mse:.6f}  "
      f"(trivial baseline: {baseline_mse:.6f}, "
      f"{'BELOW baseline -- pretext task learned something' if final_val_mse < baseline_mse else 'NOT below baseline -- investigate before trusting this pretrained trunk'})")

print("\n[SSL] Extracting pretrained trunk into a plain build_seizure_cnn_v2 "
      "shape (Dense head stays at random init -- only the trunk is "
      "pretrained; the supervised phase trains the head from scratch)...")
pretrained_trunk_model = extract_pretrained_trunk(
    best_ssl, n_channels=N_CHANNELS, window_samples=WINDOW_SAMPLES)

trunk_ckpt = 'results/pretrained_trunk_ssl_v2.h5'
pretrained_trunk_model.save(trunk_ckpt)
print(f"\nSaved: {trunk_ckpt}")
print(f"\n[SSL] Done. Next -- supervised training initialised from this trunk:")
print(f"  python3 src/models/train_baseline.py --model-version 2 "
      f"--multi-patient --init-from-ssl {trunk_ckpt} --seed 123")
