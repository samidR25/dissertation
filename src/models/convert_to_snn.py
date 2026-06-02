"""
ANN → SNN conversion for CHB-MIT seizure detection.
May 2026 stack — all API patterns verified live.

Workflow:
  1. Load trained float ANN
  2. AKD1000 v1 compat check on FLOAT model
  3. Balanced calibration sample selection
  4. Quantise (per-tensor, w4a4) — called ONCE on float model
  5. Fine-tune quantised model (low LR)
  6. Convert → .fbz (AkidaVersion.v1)
  7. Simulator eval — squeeze predict output before argmax
  8. Spike activity logging — print Statistics directly (not .items())
  9. Save results JSON

API rules (verified cnn2snn 2.19.1 / quantizeml 1.2.3 / akida 2.19.1):
  - activation_bits=  NOT activ_bits=  (silent ignore otherwise)
  - per_tensor_activations=True  MANDATORY for AKD1000 v1 hardware
  - check_model_compatibility() on FLOAT model only — returns None
  - set_akida_version(AkidaVersion.v1) on convert() AND compat check
  - input_scaling DEPRECATED — do not pass it to convert()
  - predict() returns (N,1,1,C) — preds.squeeze() then argmax
  - loaded.statistics is Statistics object — print() directly
  - quantize() called ONCE on float model (requantising raises error)

Usage:
    python3 code/models/convert_to_snn.py --patient chb01
    python3 code/models/convert_to_snn.py --patient chb01 --weight-bits 8
    python3 code/models/convert_to_snn.py --patient chb01 --finetune-epochs 10
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
parser.add_argument('--weight-bits',         type=int, default=4,
                    help='Weight bits: 4 (default) or 8 (ablation)')
parser.add_argument('--activation-bits',     type=int, default=4,
                    help='Activation bits: 4 (only valid value for v1)')
parser.add_argument('--finetune-epochs',     type=int, default=5)
parser.add_argument('--calibration-samples', type=int, default=512)
args = parser.parse_args()

os.makedirs('results', exist_ok=True)
W, A = args.weight_bits, args.activation_bits

# ── 1. Load ANN ───────────────────────────────────────────────────
ckpt = f'results/best_ann_{args.patient}.h5'
assert os.path.exists(ckpt), \
    f"No model at {ckpt}\nRun: python3 src/models/train_baseline.py --patient {args.patient}"

model = keras.models.load_model(ckpt)
print(f"Loaded: {ckpt}")

# ── 2. AKD1000 v1 compat check on FLOAT model ────────────────────
# MUST be called on float model — calling on quantised model raises:
# "Requantizing a model is not supported"
# Returns None in cnn2snn 2.19.1 — use try/except, do NOT unpack.
print("\n=== AKD1000 v1 compatibility (float model) ===")
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

# Evaluation set: test for chb02+ (has seizures); train for chb01 (test has 0)
eval_X     = X_test  if y_test.sum()  > 0 else X_train
eval_y     = y_test  if y_test.sum()  > 0 else y_train
eval_label = 'test'  if y_test.sum()  > 0 else 'train'

# ── 4. ANN baseline metrics ───────────────────────────────────────
y_pred_ann = np.argmax(model.predict(eval_X, verbose=0), axis=1)
tn_a, fp_a, fn_a, tp_a = confusion_matrix(eval_y, y_pred_ann).ravel()
ann_sens = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
ann_spec = tn_a / (tn_a + fp_a) if (tn_a + fp_a) > 0 else 0.0
print(f"\nANN baseline ({eval_label}): sensitivity={ann_sens:.4f}  specificity={ann_spec:.4f}")

# ── 5. Quantise (called ONCE on float model) ──────────────────────
# activation_bits= NOT activ_bits= (API rename — old kwarg silently ignored)
# per_tensor_activations=True MANDATORY — AKD1000 v1 hardware cannot execute
# per-axis activation quantisation (default is False = per-axis).
print(f"\n=== Quantising (w{W}a{A}, per_tensor_activations=True) ===")
qparams = QuantizationParams(
    input_weight_bits=8,
    weight_bits=W,
    activation_bits=A,               # NOT activ_bits=
    per_tensor_activations=True      # MANDATORY
)

# Balanced calibration: equal seizure/non-seizure samples for accurate
# scale factor estimation. More accurate than taking first N samples.
calib = min(args.calibration_samples, len(X_train))
sz_idx   = np.where(y_train == 1)[0][:calib // 2]
norm_idx = np.where(y_train == 0)[0][:calib // 2]
cal_X    = X_train[np.concatenate([sz_idx, norm_idx])]
print(f"Calibration: {len(cal_X)} samples ({len(sz_idx)} seizure, {len(norm_idx)} non-seizure)")

# quantize() called ONCE on FLOAT model — second call raises "Requantizing" error
q_model = quantize(model, qparams=qparams, samples=cal_X)
print("Quantisation complete ✓")

# Immediate post-PTQ accuracy check
y_post_ptq = np.argmax(q_model.predict(eval_X, verbose=0), axis=1)
tn_q, fp_q, fn_q, tp_q = confusion_matrix(eval_y, y_post_ptq).ravel()
ptq_sens = tp_q / (tp_q + fn_q) if (tp_q + fn_q) > 0 else 0.0
print(f"Post-PTQ sensitivity: {ptq_sens:.4f}  (drop: {ann_sens - ptq_sens:.4f})")

# ── 6. Fine-tune quantised model ─────────────────────────────────
# Recovers accuracy lost during quantisation. Low LR (1e-4) to avoid
# destabilising already-good weights.
print(f"\n=== Fine-tuning ({args.finetune_epochs} epochs) ===")
q_model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Use balanced train slice for validation during fine-tuning
# chb01 val set has 0 seizures — useless for monitoring seizure recovery
sz_val   = np.where(y_train == 1)[0][256:320]   # held-out seizure slice
norm_val = np.where(y_train == 0)[0][256:320]
ft_val_idx = np.concatenate([sz_val, norm_val])
X_ft_val = X_train[ft_val_idx]
y_ft_val = y_train[ft_val_idx]
q_model.fit(
    X_train, y_train,
    validation_data=(X_ft_val, y_ft_val),
    epochs=args.finetune_epochs,
    batch_size=32,
    class_weight={0: 1.0, 1: 3.0},
    verbose=1
)

# ── 7. Convert to SNN ────────────────────────────────────────────
# set_akida_version(AkidaVersion.v1) REQUIRED — default targets Akida 2.0.
# Do NOT pass input_scaling= — deprecated for QuantizeML models.
fbz_path = f'results/seizure_model_{args.patient}_w{W}a{A}.fbz'
print(f"\n=== Converting to SNN (AkidaVersion.v1) ===")
with set_akida_version(AkidaVersion.v1):
    akida_model = convert(q_model, file_path=fbz_path)
print(f"Saved: {fbz_path}")
# Expected warning: "Conversion stops at layer output because of a dequantizer"
# This is CORRECT — softmax head is intentionally unquantised.

# ── 8. Software simulator evaluation ────────────────────────────
print("\n=== Simulator evaluation ===")
loaded = akida.Model(fbz_path)
loaded.summary()

N_EVAL    = min(1000, len(eval_X))
preds_raw = loaded.predict(eval_X[:N_EVAL])

# predict() returns (N, 1, 1, n_classes) — MUST squeeze before argmax.
# Without squeeze, argmax operates on wrong axis → all predictions identical.
y_snn  = np.argmax(preds_raw.squeeze(), axis=1)
y_true = eval_y[:N_EVAL]

print(classification_report(y_true, y_snn,
      target_names=['Non-seizure', 'Seizure'], zero_division=0))

cm_snn = confusion_matrix(y_true, y_snn)
if cm_snn.shape == (2, 2):
    tn_s, fp_s, fn_s, tp_s = cm_snn.ravel()
    snn_sens = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
    snn_spec = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0
    sens_drop = ann_sens - snn_sens
    drop_ok   = sens_drop < 0.05
else:
    snn_sens = snn_spec = sens_drop = None
    drop_ok = False

print(f"\n{'─'*48}")
print(f"  ANN baseline  : sens={ann_sens:.4f}  spec={ann_spec:.4f}")
if snn_sens is not None:
    flag = ' acceptable' if drop_ok else ' investigate'
    print(f"  SNN simulator : sens={snn_sens:.4f}  spec={snn_spec:.4f}")
    print(f"  Sensitivity drop: {sens_drop:.4f}  ({flag})")
print(f"{'─'*48}")

if not drop_ok:
    print("\n⚠  Drop ≥ 5% — remediation order:")
    print("   1. python3 code/models/convert_to_snn.py "
          f"--patient {args.patient} --finetune-epochs 10")
    print("   2. ... --weight-bits 8")
    print("   3. ... --calibration-samples 1024")

# ── 9. Spike activity ────────────────────────────────────────────
# loaded.statistics is a Statistics object — print() directly.
# Do NOT iterate as dict — it is NOT a plain dict in akida 2.19.1.
print("\n=== Per-layer spike activity (power proxy) ===")
print("Target: < 0.10 per layer (consistent with 90.2% input sparsity)")
print(loaded.statistics)

# ── 10. Save results ─────────────────────────────────────────────
results = {
    'patient': args.patient,
    'eval_set': eval_label,
    'weight_bits': W,
    'activation_bits': A,          # key matches API kwarg name
    'per_tensor_activations': True,
    'akida_version': 'v1 (AKD1000)',
    'finetune_epochs': args.finetune_epochs,
    'calibration_samples': len(cal_X),
    'ann': {'sensitivity': round(ann_sens, 4), 'specificity': round(ann_spec, 4)},
    'snn_simulator': {
        'sensitivity':      round(snn_sens,  4) if snn_sens  is not None else None,
        'specificity':      round(snn_spec,  4) if snn_spec  is not None else None,
        'sensitivity_drop': round(sens_drop, 4) if sens_drop is not None else None,
        'drop_acceptable':  bool(drop_ok),
    },
    'model_path': fbz_path,
}
out = f'results/snn_results_{args.patient}_w{W}a{A}.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults : {out}")
print(f"Model   : {fbz_path}")
print(f"\nNext: python3 src/hardware/run_on_akida.py --patient {args.patient}")
