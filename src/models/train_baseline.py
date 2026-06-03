"""
Full ANN baseline training for CHB-MIT seizure detection.

Prerequisites:
  - Gate 2 smoke test passed (AKD1000 v1 compatible confirmed)
  - data/processed/{patient}_dataset_ann.npz exists

Produces:
  results/best_ann_{patient}_v{version}.h5       best checkpoint by val_loss
  results/ann_results_{patient}_v{version}.json  sensitivity, specificity, FPR/hr
  results/training_curves_{patient}_v{version}.png

Usage:
    python3 src/models/train_baseline.py --patient chb01
    python3 src/models/train_baseline.py --patient chb01 --model-version 2
    python3 src/models/train_baseline.py --patient chb01 --class-weight 1.5
    python3 src/models/train_baseline.py --patient chb01 --finetune-from chb01
"""
import argparse, json, os, sys
import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import confusion_matrix
from cnn2snn import check_model_compatibility, set_akida_version, AkidaVersion
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--patient',        default='chb01')
parser.add_argument('--model-version',  type=int, default=1, choices=[1, 2],
                    help='Architecture version: 1=v1 (1,7) temporal, '
                         '2=v2 (9,7) spatio-temporal (default: 1)')
parser.add_argument('--epochs',         type=int,   default=100)
parser.add_argument('--batch',          type=int,   default=32)
parser.add_argument('--class-weight',   type=float, default=1.5,
                    help='class_weight for seizure class (default 1.5).')
parser.add_argument('--finetune-from',  default=None,
                    help='Patient ID to load base weights from for fine-tuning. '
                         'Freezes feature extractor, trains Dense head only. '
                         'E.g. --finetune-from chb01 trains chb02 head on chb01 base.')
args = parser.parse_args()

V = args.model_version

# ── GPU ───────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"Training on GPU: {gpus[0].name}")
else:
    print("WARNING: No GPU — training will be slow")

# ── Data ──────────────────────────────────────────────────────────
npz = f'data/processed/{args.patient}_dataset_ann.npz'
assert os.path.exists(npz), \
    f"Dataset not found: {npz}\n" \
    f"Run: python3 src/preprocessing/build_dataset.py --patient {args.patient}"

data    = np.load(npz)
X_train = data['X_train'][..., np.newaxis].astype('float32')
X_val   = data['X_val']  [..., np.newaxis].astype('float32')
X_test  = data['X_test'] [..., np.newaxis].astype('float32')
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

assert X_train.shape[1:] == (18, 512, 1), \
    f"Wrong input shape {X_train.shape[1:]} — expected (18, 512, 1). " \
    f"Re-run build_dataset.py."

print(f"\nPatient : {args.patient}  |  Model: v{V}")
print(f"  Train : {len(X_train):6d}  seizure: {int(y_train.sum()):4d} ({100*y_train.mean():.1f}%)")
print(f"  Val   : {len(X_val):6d}  seizure: {int(y_val.sum()):4d}")
print(f"  Test  : {len(X_test):6d}  seizure: {int(y_test.sum()):4d}")
if y_val.sum() == 0:
    print("  NOTE: 0 seizures in val/test — all seizures in training portion.")
    print("        Sensitivity reported from training set. Cross-patient eval on chb03.")

# ── Build model ───────────────────────────────────────────────────
if V == 1:
    from src.models.akida_cnn import build_seizure_cnn
    model = build_seizure_cnn(n_channels=18, window_samples=512)
else:
    from src.models.akida_cnn_v2 import (build_seizure_cnn_v2,
                                          build_patient_adapted_model)
    if args.finetune_from:
        # Patient-specific fine-tuning: load base weights, freeze extractor
        base_ckpt = f'results/best_ann_{args.finetune_from}_v{V}.h5'
        assert os.path.exists(base_ckpt), \
            f"Base model not found: {base_ckpt}\n" \
            f"Train base first: python3 src/models/train_baseline.py " \
            f"--patient {args.finetune_from} --model-version {V}"
        base = keras.models.load_model(base_ckpt)
        model = build_patient_adapted_model(base, freeze_until='relu3')
        print(f"\nFine-tuning from: {base_ckpt}")
        print("Feature extractor frozen — training Dense head only")
    else:
        model = build_seizure_cnn_v2(n_channels=18, window_samples=512)

model.summary()

# ── Pre-training AKD1000 v1 compat check ─────────────────────────
print(f"\n=== AKD1000 v1 pre-training compatibility check (v{V}) ===")
try:
    with set_akida_version(AkidaVersion.v1):
        check_model_compatibility(model)
    print("AKD1000 v1 compatible ✓")
except Exception as e:
    print(f"INCOMPATIBLE ✗  {e}")
    sys.exit(1)

# ── Compile ───────────────────────────────────────────────────────
lr = 1e-4 if args.finetune_from else 1e-3
model.compile(
    optimizer=keras.optimizers.Adam(lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"\nLearning rate  : {lr}  {'(fine-tune)' if args.finetune_from else '(full train)'}")
print(f"class_weight   : {{0: 1.0, 1: {args.class_weight}}}")

# ── Callbacks ─────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
ckpt = f'results/best_ann_{args.patient}_v{V}.h5'

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        ckpt, monitor='val_loss',
        save_best_only=True, verbose=1
    ),
]

# ── Train ─────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=args.batch,
    class_weight={0: 1.0, 1: args.class_weight},
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ──────────────────────────────────────────────────────
model = keras.models.load_model(ckpt)

eval_X     = X_test  if y_test.sum()  > 0 else X_train
eval_y     = y_test  if y_test.sum()  > 0 else y_train
eval_label = 'test'  if y_test.sum()  > 0 else 'train'

y_pred = np.argmax(model.predict(eval_X, verbose=0), axis=1)
tn, fp, fn, tp = confusion_matrix(eval_y, y_pred).ravel()

sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
fpr_per_hour = fp / (len(eval_y) * 2 / 3600)

print(f"\n{'='*52}")
print(f"  ANN v{V} RESULTS — {args.patient} ({eval_label})")
print(f"{'='*52}")
print(f"  Sensitivity : {sensitivity:.4f}  (target ≥ 0.80)")
print(f"  Specificity : {specificity:.4f}  (target ≥ 0.90)")
print(f"  FPR/hr      : {fpr_per_hour:.2f}   (target < 2.0)")
print(f"  Epochs      : {len(history.history['loss'])}")
print(f"{'='*52}")

if sensitivity < 0.75:
    print(f"\n⚠  Sensitivity {sensitivity:.3f} below target.")
    print(f"   Retry with: --class-weight 3.0")
if fpr_per_hour > 5.0:
    print(f"\n⚠  FPR {fpr_per_hour:.2f}/hr above tolerance.")
    print(f"   Retry with: --class-weight 1.0")

# ── Save ──────────────────────────────────────────────────────────
results = {
    'patient':        args.patient,
    'model_version':  V,
    'eval_set':       eval_label,
    'sensitivity':    round(sensitivity, 4),
    'specificity':    round(specificity, 4),
    'fpr_per_hour':   round(fpr_per_hour, 2),
    'epochs_trained': len(history.history['loss']),
    'class_weight':   args.class_weight,
    'finetune_from':  args.finetune_from,
    'checkpoint':     ckpt,
    'target_met':     bool(sensitivity >= 0.75),
}
json_path = f'results/ann_results_{args.patient}_v{V}.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults : {json_path}")

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'Training v{V} — {args.patient}'
             + (f' (fine-tuned from {args.finetune_from})' if args.finetune_from else ''))
ax1.plot(history.history['loss'],     label='train')
ax1.plot(history.history['val_loss'], label='val')
ax1.set_title('Loss'); ax1.legend()
ax2.plot(history.history['accuracy'],     label='train')
ax2.plot(history.history['val_accuracy'], label='val')
ax2.set_title('Accuracy'); ax2.legend()
plt.tight_layout()
curve_path = f'results/training_curves_{args.patient}_v{V}.png'
plt.savefig(curve_path, dpi=120)
print(f"Curves  : {curve_path}")
print(f"\nNext: python3 src/models/convert_to_snn.py "
      f"--patient {args.patient} --model-version {V}")
