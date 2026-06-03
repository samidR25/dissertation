"""
ANN → SNN conversion for CHB-MIT seizure detection.

Usage:
    python3 src/models/convert_to_snn.py --patient chb01
    python3 src/models/convert_to_snn.py --patient chb01 --model-version 2
    python3 src/models/convert_to_snn.py --patient chb01 --weight-bits 2
    python3 src/models/convert_to_snn.py --patient chb01 --finetune-epochs 10

API rules (cnn2snn 2.19.1 / quantizeml 1.2.3 / akida 2.19.1):
  - activation_bits= NOT activ_bits= (silent ignore otherwise)
  - per_tensor_activations=True MANDATORY for AKD1000 v1 hardware
  - check_model_compatibility() on FLOAT model only — returns None
  - set_akida_version(AkidaVersion.v1) on convert() AND compat check
  - input_scaling DEPRECATED — do not pass it to convert()
  - predict() returns (N,1,1,C) — preds.squeeze() then argmax
  - loaded.statistics is Statistics object — print() directly
  - quantize() called ONCE on float model (requantising raises error)
"""
import argparse, json, os, sys
import numpy as np
import tensorflow as tf
import tf_keras as keras
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import (convert, check_model_compatibility,
                     set_akida_version, AkidaVersion)
from sklearn.metrics import classification_report, confusion_matrix
import akida

parser = argparse.ArgumentParser()
parser.add_argument('--patient',             default='chb01')
parser.add_argument('--model-version',       type=int, default=1, choices=[1, 2],
                    help='Architecture version: 1=v1, 2=v2 spatio-temporal')
parser.add_argument('--weight-bits',         type=int, default=4)
parser.add_argument('--activation-bits',     type=int, default=4)
parser.add_argument('--finetune-epochs',     type=int, default=5)
parser.add_argument('--calibration-samples', type=int, default=512)
args = parser.parse_args()

os.makedirs('results', exist_ok=True)
W, A, V = args.weight_bits, args.activation_bits, args.model_version

# ── 1. Load ANN ───────────────────────────────────────────────────
ckpt = f'results/best_ann_{args.patient}_v{V}.h5'

# Fallback: v1 models saved without version suffix during initial run
if not os.path.exists(ckpt) and V == 1:
    ckpt_legacy = f'results/best_ann_{args.patient}.h5'
    if os.path.exists(ckpt_legacy):
        print(f"Using legacy checkpoint: {ckpt_legacy}")
        ckpt = ckpt_legacy

assert os.path.exists(ckpt), \
    f"No model at {ckpt}\n" \
    f"Run: python3 src/models/train_baseline.py " \
    f"--patient {args.patient} --model-version {V}"

model = keras.models.load_model(ckpt)
print(f"Loaded (v{V}): {ckpt}")

# ── 2. AKD1000 v1 compat check on FLOAT model ────────────────────
print(f"\n=== AKD1000 v1 compatibility (float model v{V}) ===")
try:
    with set_akida_version(AkidaVersion.v1):
        check_model_compatibility(model)
    print("Compatible ✓")
except Exception as e:
    print(f"INCOMPATIBLE ✗  {e}")
    sys.exit(1)

# ── 3. Load data ──────────────────────────────────────────────────
data    = np.load(f'data/processed/{args.patient}_dataset_ann.npz')
X_train = data['X_train'][..., np.newaxis].astype('float32')
X_val   = data['X_val']  [..., np.newaxis].astype('float32')
X_test  = data['X_test'] [..., np.newaxis].astype('float32')
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

# Evaluation set
eval_X     = X_test  if y_test.sum()  > 0 else X_train
eval_y     = y_test  if y_test.sum()  > 0 else y_train
eval_label = 'test'  if y_test.sum()  > 0 else 'train'

# ── 4. ANN baseline metrics ───────────────────────────────────────
y_pred_ann = np.argmax(model.predict(eval_X, verbose=0), axis=1)
tn_a, fp_a, fn_a, tp_a = confusion_matrix(eval_y, y_pred_ann).ravel()
ann_sens = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
ann_spec = tn_a / (tn_a + fp_a) if (tn_a + fp_a) > 0 else 0.0
print(f"\nANN v{V} baseline ({eval_label}): "
      f"sensitivity={ann_sens:.4f}  specificity={ann_spec:.4f}")

# ── 5. Quantise (once on float model) ────────────────────────────
print(f"\n=== Quantising (w{W}a{A}, per_tensor_activations=True) ===")
qparams = QuantizationParams(
    input_weight_bits=8,
    weight_bits=W,
    activation_bits=A,
    per_tensor_activations=True
)

calib    = min(args.calibration_samples, len(X_train))
sz_idx   = np.where(y_train == 1)[0][:calib // 2]
norm_idx = np.where(y_train == 0)[0][:calib // 2]
cal_X    = X_train[np.concatenate([sz_idx, norm_idx])]
print(f"Calibration: {len(cal_X)} samples "
      f"({len(sz_idx)} seizure, {len(norm_idx)} non-seizure)")

q_model = quantize(model, qparams=qparams, samples=cal_X)
print("Quantisation complete ✓")

y_ptq = np.argmax(q_model.predict(eval_X, verbose=0), axis=1)
tn_q, fp_q, fn_q, tp_q = confusion_matrix(eval_y, y_ptq).ravel()
ptq_sens = tp_q / (tp_q + fn_q) if (tp_q + fn_q) > 0 else 0.0
print(f"Post-PTQ sensitivity: {ptq_sens:.4f}  (drop: {ann_sens - ptq_sens:.4f})")

# ── 6. Fine-tune quantised model ─────────────────────────────────
print(f"\n=== Fine-tuning ({args.finetune_epochs} epochs) ===")

# Use balanced val slice — avoids the 0-seizure val set problem
sz_val   = np.where(y_train == 1)[0][len(sz_idx):len(sz_idx)+64]
norm_val = np.where(y_train == 0)[0][len(norm_idx):len(norm_idx)+64]
ft_val_X = X_train[np.concatenate([sz_val, norm_val])]
ft_val_y = y_train[np.concatenate([sz_val, norm_val])]

q_model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
q_model.fit(
    X_train, y_train,
    validation_data=(ft_val_X, ft_val_y),
    epochs=args.finetune_epochs,
    batch_size=32,
    class_weight={0: 1.0, 1: 1.5},
    verbose=1
)

# ── 7. Convert to SNN ────────────────────────────────────────────
fbz_path = f'results/seizure_model_{args.patient}_v{V}_w{W}a{A}.fbz'
print(f"\n=== Converting to SNN (AkidaVersion.v1) ===")
with set_akida_version(AkidaVersion.v1):
    akida_model = convert(q_model, file_path=fbz_path)
print(f"Saved: {fbz_path}")

# ── 8. Simulator evaluation ───────────────────────────────────────
print("\n=== Simulator evaluation ===")
loaded = akida.Model(fbz_path)
loaded.summary()

# Balanced eval slice — avoids 0-seizure first-N-windows problem
sz_idx_e   = np.where(eval_y == 1)[0]
norm_idx_e = np.where(eval_y == 0)[0]

if len(sz_idx_e) > 0:
    n_each   = min(500, len(sz_idx_e), len(norm_idx_e))
    eval_idx = np.sort(np.concatenate([sz_idx_e[:n_each],
                                        norm_idx_e[:n_each]]))
else:
    eval_idx = np.arange(min(1000, len(eval_y)))

preds_raw = loaded.predict(eval_X[eval_idx])
y_snn     = np.argmax(preds_raw.squeeze(), axis=1)
y_true    = eval_y[eval_idx]

print(classification_report(y_true, y_snn,
      target_names=['Non-seizure', 'Seizure'], zero_division=0))

cm_snn = confusion_matrix(y_true, y_snn)
if cm_snn.shape == (2, 2):
    tn_s, fp_s, fn_s, tp_s = cm_snn.ravel()
    snn_sens  = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
    snn_spec  = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0
    sens_drop = ann_sens - snn_sens
    drop_ok   = sens_drop < 0.05
else:
    snn_sens = snn_spec = sens_drop = None
    drop_ok = False

print(f"\n{'─'*50}")
print(f"  ANN v{V} baseline : sens={ann_sens:.4f}  spec={ann_spec:.4f}")
if snn_sens is not None:
    flag = '✓ acceptable' if drop_ok else '✗ investigate'
    print(f"  SNN simulator  : sens={snn_sens:.4f}  spec={snn_spec:.4f}")
    print(f"  Sensitivity drop: {sens_drop:.4f}  ({flag})")
print(f"{'─'*50}")

if not drop_ok and snn_sens is not None:
    print(f"\n⚠  Drop ≥ 5% — try:")
    print(f"   --finetune-epochs 10  or  --calibration-samples 1024")

# ── 9. Spike activity ─────────────────────────────────────────────
print("\n=== Per-layer spike activity (power proxy) ===")
print("Target: < 0.10 per layer")
print(loaded.statistics)

# ── 10. Save results ──────────────────────────────────────────────
results = {
    'patient':           args.patient,
    'model_version':     V,
    'eval_set':          eval_label,
    'weight_bits':       W,
    'activation_bits':   A,
    'per_tensor':        True,
    'akida_version':     'v1 (AKD1000)',
    'finetune_epochs':   args.finetune_epochs,
    'calibration_samples': len(cal_X),
    'ann': {
        'sensitivity': round(ann_sens, 4),
        'specificity': round(ann_spec, 4),
    },
    'snn_simulator': {
        'sensitivity':      round(snn_sens,  4) if snn_sens  is not None else None,
        'specificity':      round(snn_spec,  4) if snn_spec  is not None else None,
        'sensitivity_drop': round(sens_drop, 4) if sens_drop is not None else None,
        'drop_acceptable':  bool(drop_ok),
    },
    'model_path': fbz_path,
}
out = f'results/snn_results_{args.patient}_v{V}_w{W}a{A}.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults : {out}")
print(f"Model   : {fbz_path}")
print(f"\nNext: python3 src/hardware/run_on_akida.py "
      f"--patient {args.patient} --model-version {V}")
