"""
Gate-based smoke test for CHB-MIT seizure model.
Run before every full training session.

Gates:
  1  200 samples,  2 epochs  — pipeline intact; no NaN; loss decreasing
  2  2000 samples, 5 epochs  — AKD1000 v1 compatible; sensitivity > random
  3  10000 samples,15 epochs — val loss drops; overfit gap < 0.30

Usage:
    python3 code/models/smoke_test.py --gate 1
    python3 code/models/smoke_test.py --gate 2
    python3 code/models/smoke_test.py --gate 3

API rules (cnn2snn 2.19.1 / quantizeml 1.2.3 — verified live):
  - check_model_compatibility() returns None — try/except, no unpacking
  - Call on FLOAT model only
  - Wrap in set_akida_version(AkidaVersion.v1)
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
import argparse, sys
import numpy as np
import tensorflow as tf
import tf_keras as keras
from cnn2snn import check_model_compatibility, set_akida_version, AkidaVersion

parser = argparse.ArgumentParser()
parser.add_argument('--gate',    type=int, default=1, choices=[1, 2, 3])
parser.add_argument('--patient', default='chb01')
parser.add_argument('--model-version', type=int, default=2, choices=[1, 2])
args = parser.parse_args()

GATE_CONFIG = {
    1: {'samples': 200,   'epochs': 2},
    2: {'samples': 2000,  'epochs': 5},
    3: {'samples': 10000, 'epochs': 15},
}
cfg = GATE_CONFIG[args.gate]

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
print(f"Device: {'GPU' if gpus else 'CPU'} | Gate: {args.gate} | "
      f"Samples: {cfg['samples']} | Epochs: {cfg['epochs']}")

# ── Load data ─────────────────────────────────────────────────────
data  = np.load(f'data/processed/{args.patient}_dataset_ann.npz')
n     = cfg['samples']
n_val = max(50, n // 4)

X_tr = data['X_train'][:n,     ..., np.newaxis].astype('float32')
y_tr = data['y_train'][:n]
X_v  = data['X_val'][:n_val,   ..., np.newaxis].astype('float32')
y_v  = data['y_val'][:n_val]

print(f"Train: {len(X_tr)} | Val: {len(X_v)} | "
      f"Seizure in train: {int(y_tr.sum())} | in val: {int(y_v.sum())}")
# ── Build model ───────────────────────────────────────────────────
if args.model_version == 1:
    from src.models.akida_cnn import build_seizure_cnn
    model = build_seizure_cnn(n_channels=18, window_samples=512)
else:
    from src.models.akida_cnn_v2 import build_seizure_cnn_v2
    model = build_seizure_cnn_v2(n_channels=18, window_samples=512)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── Train ─────────────────────────────────────────────────────────
hist = model.fit(
    X_tr, y_tr,
    validation_data=(X_v, y_v),
    epochs=cfg['epochs'],
    batch_size=32,
    class_weight={0: 1.0, 1: 3.0},
    verbose=0
)
losses = hist.history['loss']

# ── Gate 1 ────────────────────────────────────────────────────────
if args.gate >= 1:
    if any(np.isnan(l) for l in losses):
        print("Gate 1 ✗  NaN loss — check input dtype and data pipeline")
        sys.exit(1)
    if losses[-1] >= losses[0]:
        print(f"Gate 1 ✗  Loss not decreasing: {losses[0]:.4f} → {losses[-1]:.4f}")
        print("         Check: class_weight, LR, input normalisation")
        sys.exit(1)
    print(f"Gate 1 ✓  Loss {losses[0]:.4f} → {losses[-1]:.4f}  (pipeline clean)")

# ── Gate 2 ────────────────────────────────────────────────────────
if args.gate >= 2:
    # check_model_compatibility() returns None in cnn2snn 2.19.1.
    # Call on FLOAT model only. Wrap in set_akida_version(AkidaVersion.v1).
    print("\nChecking AKD1000 v1 compatibility...")
    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(model)
        print("Gate 2 ✓  AKD1000 v1 compatible")
    except Exception as e:
        print(f"Gate 2 ✗  INCOMPATIBLE: {e}")
        print("         Fix architecture in akida_cnn.py — do not proceed to training")
        sys.exit(1)

    # Sensitivity check — skip for chb01 (0 seizures in val set by design)
    y_pred = np.argmax(model.predict(X_v, verbose=0), axis=1)
    tp = int(((y_pred == 1) & (y_v == 1)).sum())
    fn = int(((y_pred == 0) & (y_v == 1)).sum())
    if (tp + fn) == 0:
        print("Gate 2 ✓  Sensitivity check skipped — 0 seizures in val "
              "(expected for chb01 chronological split)")
    else:
        sens = tp / (tp + fn)
        if sens < 0.30:
            print(f"Gate 2 ✗  Sensitivity {sens:.3f} — model not learning seizures")
            print("         Increase class_weight seizure ratio or check SMOTE output")
            sys.exit(1)
        print(f"Gate 2 ✓  Sensitivity {sens:.3f}")

    print("\nGate 2 PASSED — safe to run full training")

# ── Gate 3 ────────────────────────────────────────────────────────
if args.gate >= 3:
    vl = hist.history['val_loss']
    if vl[-1] >= vl[0]:
        print(f"Gate 3 ✗  Val loss not dropping: {vl[0]:.4f} → {vl[-1]:.4f}")
        sys.exit(1)

    train_acc = hist.history['accuracy'][-1]
    val_acc   = hist.history['val_accuracy'][-1]
    gap       = train_acc - val_acc

    if gap > 0.30:
        print(f"Gate 3 ✗  Overfit gap {gap:.3f} (train={train_acc:.3f}, val={val_acc:.3f})")
        print("         Add Dropout(0.3) after relu3 in akida_cnn.py (between blocks — probe3 E confirmed)")
        sys.exit(1)

    print(f"Gate 3 ✓  Val loss {vl[0]:.4f} → {vl[-1]:.4f} | Overfit gap {gap:.3f}")
    print("\nGate 3 PASSED → run full training:")
    print("   python3 src/models/train_baseline.py --patient chb01")
