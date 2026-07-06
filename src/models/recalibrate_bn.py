"""
src/models/recalibrate_bn.py
=============================
Item 4 (BN-only recalibration) — cheapest extension of the Items 2/3
calibration family. No gradient descent, no optimizer, no weight updates
of any kind: this script only overwrites the moving_mean/moving_variance
of bn1/bn2/bn3 with statistics computed from a new patient's own short
calibration window, then hands off to the existing convert_to_snn.py
pipeline unchanged.

Why momentum=0 instead of N epochs of EMA:
    Keras BatchNormalization in training mode updates
        moving_mean = momentum * moving_mean + (1 - momentum) * batch_mean
    With momentum left at its default (~0.99), a single forward pass barely
    moves the running stats, and "how many passes is enough" has no
    principled answer. Setting momentum=0.0 for exactly one forward pass
    makes moving_mean/moving_variance EXACTLY equal to the calibration
    batch's own mean/var -- deterministic, no tuning, no ambiguity about
    convergence. Momentum is restored to its original value before saving
    (cosmetic correctness only, since no further training happens).

Scaling convention -- deliberately NOT composed with Item 2:
    The multi-pool base checkpoint's manifest stores a `per_patient`
    scaler dict (one scaler per pool constituent: chb01/02/05), not a
    single flat scaler -- there is no canonical "pool scaler" a held-out
    patient could be recalibrated against. Every C1 eval run so far uses
    the held-out patient's OWN full-recording scaler
    (data/processed/{patient}_scaler.json) by default, and this script
    keeps that convention so BN recalibration's effect can be read in
    isolation, one variable at a time, before any future session tries
    combining it with Item 2's calibration-window rescaling.

Usage:
    python3 src/models/recalibrate_bn.py --patient chb13
    python3 src/models/recalibrate_bn.py --patient chb16 --calib-seconds 180

Output: results/best_ann_multi_bnrecal_{patient}_v2.h5 (+ manifest)
Next:   python3 src/models/convert_to_snn.py \\
            --base multi_bnrecal_{patient} --patient {patient} \\
            --eval-patient {patient} --model-version 2
        (reuses the existing, already-correct quantise/convert/eval
        pipeline unchanged -- wrap, don't replace)
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

import argparse
import json
import sys
import numpy as np
import tf_keras as keras

sys.path.insert(0, '.')
from src.manifest import write_manifest, load_manifest

parser = argparse.ArgumentParser()
parser.add_argument('--patient', required=True,
                     help="Held-out patient to recalibrate BN statistics for.")
parser.add_argument('--base-tag', default='multi',
                     help="Frozen-pool base checkpoint tag (default 'multi' "
                          "-> results/best_ann_multi_v2.h5)")
parser.add_argument('--model-version', type=int, default=2)
parser.add_argument('--calib-seconds', type=int, default=120,
                     help="Length of calibration window in seconds, same "
                          "convention as compute_calib_scaler.py (Item 2). "
                          "Window stride is ~1.0s, so window index i starts "
                          "at approximately t=i seconds.")
parser.add_argument('--input-dir', default='data/processed/')
args = parser.parse_args()

ckpt_path = f'results/best_ann_{args.base_tag}_v{args.model_version}.h5'
print(f"Base checkpoint : {ckpt_path}")
print(f"Target patient  : {args.patient}")

if not os.path.exists(ckpt_path):
    sys.exit(f"ERROR: base checkpoint not found: {ckpt_path}")

# ── 1. Load raw calibration window (first N seconds, non-seizure) ───────────
X_path = os.path.join(args.input_dir, f'{args.patient}_X.npy')
y_path = os.path.join(args.input_dir, f'{args.patient}_y.npy')
assert os.path.exists(X_path) and os.path.exists(y_path), \
    f"Raw windows not found: {X_path} / {y_path}\n" \
    f"Regenerate with: python3 src/preprocessing/preprocess.py --patient {args.patient}"

X_raw = np.load(X_path)
y_raw = np.load(y_path)
n_calib = min(args.calib_seconds, X_raw.shape[0])
X_calib_raw = X_raw[:n_calib]
y_calib = y_raw[:n_calib]

n_seizure_in_calib = int(y_calib.sum())
if n_seizure_in_calib > 0:
    raise SystemExit(
        f"REFUSING: first {args.calib_seconds}s of {args.patient}'s recording "
        f"contains {n_seizure_in_calib} seizure window(s). Same hard "
        f"constraint as compute_calib_scaler.py -- a calibration window "
        f"must be seizure-free by construction. Try a different "
        f"--calib-seconds to dodge an early seizure."
    )
print(f"Calibration window : first {args.calib_seconds}s "
      f"({n_calib} windows), confirmed seizure-free")

# ── 2. Scale using the patient's OWN full-recording scaler ──────────────────
# (NOT Item 2's calibration-window scaler -- see module docstring for why)
scaler_path = os.path.join(args.input_dir, f'{args.patient}_scaler.json')
assert os.path.exists(scaler_path), f"Scaler not found: {scaler_path}"
with open(scaler_path) as f:
    own_scaler = json.load(f)
scale, shift = own_scaler['scale'], own_scaler['shift']
print(f"Scaling under : {scaler_path} (scale={scale:.4f}, shift={shift:.4f})")

X_calib = (X_calib_raw * scale + shift).clip(0, 255).astype('float32')
X_calib = X_calib[..., np.newaxis]  # add channel dim -> (N, 18, 512, 1)
assert X_calib.shape[1:] == (18, 512, 1), f"Unexpected shape {X_calib.shape}"

# ── 3. Load float model, locate BN layers ────────────────────────────────────
print(f"\nLoading float model: {ckpt_path}")
model = keras.models.load_model(ckpt_path)

bn_names = ['bn1', 'bn2', 'bn3']
bn_layers = []
for name in bn_names:
    layer = model.get_layer(name)
    bn_layers.append(layer)
    print(f"  Found {name}: momentum={layer.momentum}, "
          f"moving_mean[:3]={layer.moving_mean.numpy().flatten()[:3]}")

# ── 4. One deterministic forward pass with momentum=0 ────────────────────────
original_momenta = [l.momentum for l in bn_layers]
for l in bn_layers:
    l.momentum = 0.0

print(f"\nRunning one forward pass (training=True, momentum=0) over "
      f"{len(X_calib)} calibration windows...")
_ = model(X_calib, training=True)

for l, m in zip(bn_layers, original_momenta):
    l.momentum = m  # restore -- cosmetic, no further training happens

print("\nPost-recalibration BN stats:")
for name, layer in zip(bn_names, bn_layers):
    print(f"  {name}: moving_mean[:3]={layer.moving_mean.numpy().flatten()[:3]}, "
          f"moving_variance[:3]={layer.moving_variance.numpy().flatten()[:3]}")

# ── 5. Save recalibrated checkpoint + manifest ───────────────────────────────
save_tag = f'{args.base_tag}_bnrecal_{args.patient}'
save_path = f'results/best_ann_{save_tag}_v{args.model_version}.h5'
model.save(save_path)
print(f"\nSaved: {save_path}")

base_manifest = load_manifest(ckpt_path, required=False)
write_manifest(
    save_path,
    source_checkpoint=ckpt_path,
    base_manifest_scaler=(base_manifest.get('scaler') if base_manifest else None),
    bn_recalibrated_on=args.patient,
    calib_seconds=args.calib_seconds,
    n_calib_windows=n_calib,
    recalibration_method='momentum=0, single forward pass, training=True',
    scaler={'scale': scale, 'shift': shift},  # this checkpoint now expects
                                                # the patient's own scaler --
                                                # matches convert_to_snn.py's
                                                # default consistency check
)
print(f"Manifest saved: {save_path}.manifest.json")

print(f"\nNext: python3 src/models/convert_to_snn.py \\\n"
      f"    --base {save_tag} --patient {args.patient} \\\n"
      f"    --eval-patient {args.patient} --model-version {args.model_version}")
