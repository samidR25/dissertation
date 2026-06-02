"""
Full ANN baseline training for CHB-MIT seizure detection.

Prerequisites:
  - Gate 2 smoke test passed (AKD1000 v1 compatible confirmed)
  - data/processed/{patient}_dataset_ann.npz exists

Produces:
  results/best_ann_{patient}.h5          best checkpoint by val_loss
  results/ann_results_{patient}.json     sensitivity, specificity, FPR/hr
  results/training_curves_{patient}.png  loss + accuracy curves

Usage:
    python3 code/models/train_baseline.py
    python3 code/models/train_baseline.py --patient chb02
    python3 code/models/train_baseline.py --patient chb01 --class-weight 5.0
    python3 code/models/train_baseline.py --patient chb01 --epochs 150 --batch 32
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
parser.add_argument('--patient',      default='chb01')
parser.add_argument('--epochs',       type=int,   default=100)
parser.add_argument('--batch',        type=int,   default=32)
parser.add_argument('--class-weight', type=float, default=3.0,
                    help='class_weight for seizure class (default 3.0). '
                         'Increase to 5.0 if sensitivity < 0.75. '
                         'Decrease to 2.0 if FPR > 5/hr.')
args = parser.parse_args()

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
    f"Dataset not found: {npz}\nRun: python3 code/preprocessing/build_dataset.py"

data    = np.load(npz)
X_train = data['X_train'][..., np.newaxis].astype('float32')
X_val   = data['X_val']  [..., np.newaxis].astype('float32')
X_test  = data['X_test'] [..., np.newaxis].astype('float32')
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

assert X_train.shape[1:] == (18, 512, 1), \
    f"Wrong input shape {X_train.shape[1:]} — expected (18, 512, 1). Re-run build_dataset.py."

print(f"\nPatient : {args.patient}")
print(f"  Train : {len(X_train):6d}  seizure: {int(y_train.sum()):4d} ({100*y_train.mean():.1f}%)")
print(f"  Val   : {len(X_val):6d}  seizure: {int(y_val.sum()):4d}")
print(f"  Test  : {len(X_test):6d}  seizure: {int(y_test.sum()):4d}")
if y_val.sum() == 0:
    print("  NOTE: 0 seizures in val/test — expected for chb01.")
    print("        Sensitivity reported from training set for chb01 only.")

# ── Build + pre-training compat check ────────────────────────────
from src.models.akida_cnn import build_seizure_cnn

model = build_seizure_cnn(n_channels=18, window_samples=512)
model.summary()

# Pre-training compat check: catches architecture issues before wasting GPU time.
# check_model_compatibility() returns None in cnn2snn 2.19.1 — use try/except.
print("\n=== AKD1000 v1 pre-training compatibility check ===")
try:
    with set_akida_version(AkidaVersion.v1):
        check_model_compatibility(model)
    print("AKD1000 v1 compatible ✓")
except Exception as e:
    print(f"INCOMPATIBLE ✗  {e}")
    sys.exit(1)

# ── Compile ───────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ─────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
ckpt = f'results/best_ann_{args.patient}.h5'

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
print(f"\nclass_weight = {{0: 1.0, 1: {args.class_weight}}}")
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

# Use training set for chb01 (0 seizures in test); test set for other patients
eval_X = X_test  if y_test.sum()  > 0 else X_train
eval_y = y_test  if y_test.sum()  > 0 else y_train
eval_label = 'test' if y_test.sum() > 0 else 'train (chb01: no seizures in test)'

y_pred = np.argmax(model.predict(eval_X, verbose=0), axis=1)
tn, fp, fn, tp = confusion_matrix(eval_y, y_pred).ravel()

sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
fpr_per_hour = fp / (len(eval_y) * 2 / 3600)   # FP windows × 2s / 3600s

print(f"\n{'='*50}")
print(f"  ANN RESULTS — {args.patient} ({eval_label})")
print(f"{'='*50}")
print(f"  Sensitivity : {sensitivity:.4f}  (target ≥ 0.80)")
print(f"  Specificity : {specificity:.4f}  (target ≥ 0.90)")
print(f"  FPR/hr      : {fpr_per_hour:.2f}   (target < 2.0)")
print(f"  Epochs      : {len(history.history['loss'])}")
print(f"{'='*50}")

if sensitivity < 0.75:
    print(f"\n⚠  Sensitivity {sensitivity:.3f} below target.")
    print(f"   Retry: python3 code/models/train_baseline.py --patient {args.patient} --class-weight 5.0")
if fpr_per_hour > 5.0:
    print(f"\n⚠  FPR {fpr_per_hour:.2f}/hr above tolerance.")
    print(f"   Retry: python3 code/models/train_baseline.py --patient {args.patient} --class-weight 2.0")

# ── Save ──────────────────────────────────────────────────────────
results = {
    'patient': args.patient, 'eval_set': eval_label,
    'sensitivity': round(sensitivity, 4),
    'specificity': round(specificity, 4),
    'fpr_per_hour': round(fpr_per_hour, 2),
    'epochs_trained': len(history.history['loss']),
    'class_weight': args.class_weight,
    'checkpoint': ckpt,
    'target_met': bool(sensitivity >= 0.75),
}
json_path = f'results/ann_results_{args.patient}.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults: {json_path}")

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'Training — {args.patient}')
ax1.plot(history.history['loss'],     label='train')
ax1.plot(history.history['val_loss'], label='val')
ax1.set_title('Loss'); ax1.legend()
ax2.plot(history.history['accuracy'],     label='train')
ax2.plot(history.history['val_accuracy'], label='val')
ax2.set_title('Accuracy'); ax2.legend()
plt.tight_layout()
curve_path = f'results/training_curves_{args.patient}.png'
plt.savefig(curve_path, dpi=120)
print(f"Curves:  {curve_path}")
print(f"\nNext: python3 code/models/convert_to_snn.py --patient {args.patient}")
