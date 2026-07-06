"""
src/models/convert_to_snn.py
==============================
Quantise and convert ANN → SNN (.fbz) for AKD1000 v1.

Phase 2a usage (unchanged):
    python3 src/models/convert_to_snn.py --patient chb01 --model-version 2
    python3 src/models/convert_to_snn.py --patient chb03 --model-version 2

Phase 2c additions:
    # Convert multi-patient base model (saves seizure_model_multi_v2_w4a4.fbz)
    python3 src/models/convert_to_snn.py --base multi --model-version 2

    # Convert chb03 gradual-unfreeze fine-tuned model (same as Phase 2a — no flag needed)
    python3 src/models/convert_to_snn.py --patient chb03 --model-version 2

Flags:
    --patient       chbXX   Patient tag for .h5 input and .npz calibration data.
                            Used as default base tag if --base not set.
    --base          TAG     Override ANN checkpoint base tag independently of
                            --patient.  E.g. --base multi loads
                            results/best_ann_multi_v2.h5 while using
                            --patient chb03 data for calibration.
    --model-version 1|2     Architecture version (default 2).
    --w-bits        4       Weight quantisation bits (default 4).
    --a-bits        4       Activation quantisation bits (default 4).
    --cal-samples   256     Number of calibration samples for PTQ.

Output: results/seizure_model_<base>_v<V>_w<W>a<A>.fbz
        results/snn_results_<base>_v<V>_w<W>a<A>.json

Non-negotiable rules (AKD1000 v1):
    • import tf_keras as keras                — never tensorflow.keras
    • set_akida_version(AkidaVersion.v1)      — wraps every convert() call
    • QuantizationParams(per_tensor_activations=True)  — mandatory for v1 silicon
    • check_model_compatibility() on float model only, returns None, try/except
    • predict() output is (N,1,1,C) — squeeze then argmax
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
import tf_keras as keras
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import (check_model_compatibility, convert,
                      set_akida_version, AkidaVersion)
from sklearn.metrics import confusion_matrix
import akida
sys.path.insert(0, '.')
from src.manifest import write_manifest, load_manifest

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--patient',       default='chb01',
                    help="Patient tag for calibration data and default eval dataset.")
parser.add_argument('--base',          default=None,
                    help="Override ANN checkpoint tag. "
                         "E.g. 'multi' → results/best_ann_multi_v2.h5. "
                         "If not set, --patient is used.")
parser.add_argument('--variant',       default=None,
                    help="Checkpoint variant suffix, e.g. 'noz' or 'zscore'. "
                         "results/best_ann_<base>_v<V>_<variant>.h5. "
                         "If not set, loads results/best_ann_<base>_v<V>.h5 (no suffix).")
parser.add_argument('--eval-patient',  default=None,
                    help="Patient tag to evaluate the SNN on, independent of "
                         "calibration data. E.g. --base multi --eval-patient chb03 "
                         "calibrates on multi data, evaluates on chb03's test set. "
                         "If not set, falls back to --patient.")
parser.add_argument('--model-version', type=int, default=2)
parser.add_argument('--w-bits',        type=int, default=4)
parser.add_argument('--a-bits',        type=int, default=4)
parser.add_argument('--cal-samples',   type=int, default=256)
# add to the argparse block, near the other arguments:
parser.add_argument('--seed', type=int, default=42,
                    help="Random seed for PTQ calibration determinism. "
                         "convert_to_snn.py previously had no determinism "
                         "controls — calibration could vary run-to-run via "
                         "non-deterministic GPU kernels, producing "
                         "meaningfully different quantized models from the "
                         "identical checkpoint + calibration data (found "
                         "during Condition 4 pilot, Gate 3i).")
parser.add_argument('--longctx',        action='store_true',
                    help='Use 3-channel long-context dataset for PTQ '
                         'calibration (Gate 2 Arms B/C).')
parser.add_argument('--window-samples', type=int, default=512,
                    choices=[512, 768],
                    help='Window size in samples. Only meaningful with --longctx.')
parser.add_argument('--g-features',     action='store_true',
                    help='Use Candidate G relative-band-power dataset for '
                         'PTQ calibration/eval (Handoff_a_e_c_closed_to_'
                         'next_steps.md sec8 phase 1). Mutually exclusive '
                         'with --longctx. NOTE: pass --patient multi (not '
                         'multi_g) for calibration-data resolution -- use '
                         '--base multi_g to load the checkpoint, same '
                         'decoupling already used for --coral/--dann via '
                         '--variant.')
args = parser.parse_args()
# immediately after args = parser.parse_args():
import random
random.seed(args.seed)
np.random.seed(args.seed)
import tensorflow as tf
tf.random.set_seed(args.seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
print(f"Random seed: {args.seed}")

# Resolve which checkpoint to load
base_tag    = args.base if args.base else args.patient
eval_tag    = args.eval_patient if args.eval_patient else args.patient
variant_sfx = f'_{args.variant}' if args.variant else ''

ckpt_path = f'results/best_ann_{base_tag}_v{args.model_version}{variant_sfx}.h5'

# fbz/json filenames include eval_tag when it differs from base_tag, so
# converting the same base model against multiple eval patients doesn't
# silently overwrite results (e.g. multi → chb03 vs multi → chb10)
tag_for_outputs = base_tag if eval_tag == base_tag else f'{base_tag}_on_{eval_tag}'
fbz_path  = (f'results/seizure_model_{base_tag}{variant_sfx}'
             f'_v{args.model_version}_w{args.w_bits}a{args.a_bits}.fbz')
json_path = (f'results/snn_results_{tag_for_outputs}{variant_sfx}'
             f'_v{args.model_version}_w{args.w_bits}a{args.a_bits}.json')

print(f"ANN checkpoint : {ckpt_path}")
print(f"Calibration    : data/processed/{args.patient}_dataset_ann.npz "
      f"({args.cal_samples} samples)")
print(f"Eval patient   : {eval_tag}")
print(f"Output SNN     : {fbz_path}")
os.makedirs('results', exist_ok=True)

if not os.path.exists(ckpt_path):
    sys.exit(
        f"ERROR: checkpoint not found: {ckpt_path}\n"
        "Either train the model first:\n"
        f"  python3 src/models/train_baseline.py "
        f"--model-version {args.model_version} "
        + (f"--multi-patient" if base_tag == 'multi'
           else f"--patient {base_tag}")
        + "\nor check --variant matches an existing suffix "
          "(e.g. --variant noz for best_ann_multi_v2_noz.h5)."
    )

ckpt_manifest = load_manifest(ckpt_path, required=False)
if ckpt_manifest is None:
    print(f"  WARNING: no manifest for {ckpt_path} — this checkpoint predates "
          "Gate 1b. The .fbz produced here will carry no scaler provenance "
          "either; regenerate the checkpoint to fix this.")
elif ckpt_manifest.get('scaler') and 'per_patient' not in ckpt_manifest['scaler']:
    if args.longctx or args.g_features:
        # Longctx/G scalers use per-channel {ch0_min,...} keys, not flat
        # {scale,shift}. Skip the flat-format consistency check — Gate 2c
        # in train_baseline.py already verified the correct scaler was used.
        _tag = 'Longctx' if args.longctx else 'Candidate G'
        print(f"  [Gate 1b] {_tag} checkpoint — per-channel scaler format; "
              "flat-format consistency check skipped (Gate 2c verified at training).")
    else:
        cal_scaler_path = f'data/processed/{args.patient}_scaler.json'
        if os.path.exists(cal_scaler_path):
            with open(cal_scaler_path) as f:
                cal_scaler = json.load(f)
            m = ckpt_manifest['scaler']
            if (abs(m['scale'] - cal_scaler['scale']) > 1e-3 or
                    abs(m['shift'] - cal_scaler['shift']) > 1e-3):
                print(f"  WARNING: calibration data ({cal_scaler_path}) scaler "
                      f"differs from {ckpt_path}'s training scaler "
                      f"(model: scale={m['scale']:.2f}/shift={m['shift']:.2f}, "
                      f"calibration: scale={cal_scaler['scale']:.2f}/"
                      f"shift={cal_scaler['shift']:.2f}). Quantisation will "
                      "calibrate against a different input distribution than "
                      "the model was trained on — verify this is intentional.")

# ── 1. Load float model ────────────────────────────────────────────────────────
print(f"\nLoading float model: {ckpt_path}")
float_model = keras.models.load_model(ckpt_path)
float_model.summary(print_fn=lambda x: print(f"  {x}"))

# ── 2. AKD1000 v1 compatibility check (on float model ONLY) ──────────────────
print("\nChecking AKD1000 v1 compatibility...")
try:
    with set_akida_version(AkidaVersion.v1):
        check_model_compatibility(float_model)
    print("AKD1000 v1 compatible ✓")
except Exception as e:
    # check_model_compatibility returns None in cnn2snn 2.19.1 — may be a
    # false alarm from the context manager; proceed with caution
    print(f"  WARNING: compatibility check raised: {e}")
    print("  Proceeding — if convert() fails, check architecture constraints.")

# ── 3. Load calibration data ──────────────────────────────────────────────────
if args.longctx and args.g_features:
    sys.exit("ERROR: --longctx and --g-features are mutually exclusive.")
if args.longctx:
    data_path = (f'data/processed/{args.patient}_dataset_longctx_w'
                 f'{args.window_samples}.npz')
elif args.g_features:
    data_path = f'data/processed/{args.patient}_dataset_g.npz'
else:
    data_path = f'data/processed/{args.patient}_dataset_ann.npz'
if not os.path.exists(data_path):
    if args.longctx:
        _hint = (f"Run: python3 src/preprocessing/build_dataset_longctx.py "
                f"--patient {args.patient} --window-samples {args.window_samples}")
    elif args.g_features:
        _hint = (f"Run: python3 src/preprocessing/build_dataset_g.py "
                f"--patient {args.patient}  (or --multi-patient, then pass "
                f"--patient multi here)")
    else:
        _hint = f"Run: python3 src/preprocessing/build_dataset.py --patient {args.patient}"
    sys.exit(f"ERROR: {data_path} not found.\n{_hint}")

data    = np.load(data_path)
if args.longctx or args.g_features:
    X_train = data['X_train'].astype('float32')
else:
    X_train = data['X_train'][..., np.newaxis].astype('float32')
y_train = data['y_train']

# Balanced calibration slice: equal seizure / non-seizure from training set.
# This gives PTQ recovery a meaningful signal even when seizures are rare.
seiz_idx = np.where(y_train == 1)[0]
nons_idx = np.where(y_train == 0)[0]
half_cal = args.cal_samples // 2
s_idx    = seiz_idx[:half_cal]
n_idx    = nons_idx[:half_cal]

if len(s_idx) == 0:
    print("WARNING: no seizure windows in training set — using first "
          f"{args.cal_samples} windows for calibration")
    X_cal = X_train[:args.cal_samples]
else:
    cal_idx = np.concatenate([s_idx, n_idx])
    X_cal   = X_train[cal_idx]

print(f"\nCalibration set: {len(X_cal)} windows "
      f"(seizure: {int(y_train[cal_idx if len(s_idx)>0 else np.arange(args.cal_samples)].sum())})")
if args.longctx:
    assert X_cal.shape[1:] == (18, args.window_samples, 3), \
        f"Wrong calibration shape {X_cal.shape}"
elif args.g_features:
    assert X_cal.shape[1:] == (18, 512, 3), \
        f"Wrong calibration shape {X_cal.shape}"
else:
    assert X_cal.shape[1:] == (18, 512, 1), \
        f"Wrong calibration shape {X_cal.shape}"

# ── 4. Quantise ───────────────────────────────────────────────────────────────
print(f"\nQuantising (w{args.w_bits}a{args.a_bits}, per_tensor_activations=True)...")
qparams = QuantizationParams(
    input_weight_bits=8,          # input layer always 8-bit
    weight_bits=args.w_bits,
    activation_bits=args.a_bits,  # NOT activ_bits= — that kwarg is silently ignored
    per_tensor_activations=True   # MANDATORY for AKD1000 v1 hardware
)
q_model = quantize(float_model, qparams=qparams, samples=X_cal)
print("Quantisation complete ✓")

# ── 5. Convert to SNN ─────────────────────────────────────────────────────────
print(f"\nConverting to SNN (AkidaVersion.v1)...")
with set_akida_version(AkidaVersion.v1):
    akida_model = convert(q_model, file_path=fbz_path)
write_manifest(
    fbz_path,
    source_checkpoint=ckpt_path,
    scaler=(ckpt_manifest.get('scaler') if ckpt_manifest else None),
    base_tag=base_tag,
    calibration_patient=args.patient,
    model_version=args.model_version,
    weight_bits=args.w_bits,
    activation_bits=args.a_bits,
    cal_samples=len(X_cal),
    seed=args.seed,
)
print(f"Manifest saved: {fbz_path}.manifest.json")
print(f"SNN saved: {fbz_path}")

# ── 6. SNN evaluation on test set ────────────────────────────────────────────
# Evaluate on eval_tag's data, NOT necessarily the calibration patient's data.
# This is what lets --base multi --eval-patient chb03 calibrate on multi
# training data but evaluate cross-patient generalisation on chb03's actual
# chronological test split.
if eval_tag != args.patient:
    if args.g_features:
        eval_data_path = f'data/processed/{eval_tag}_dataset_g.npz'
        _eval_hint = (f"Run: python3 src/preprocessing/build_dataset_g.py "
                     f"--patient {eval_tag}")
    else:
        eval_data_path = f'data/processed/{eval_tag}_dataset_ann.npz'
        _eval_hint = (f"Run: python3 src/preprocessing/build_dataset.py "
                     f"--patient {eval_tag}")
    if not os.path.exists(eval_data_path):
        sys.exit(f"ERROR: {eval_data_path} not found.\n{_eval_hint}")
    eval_data = np.load(eval_data_path)
else:
    eval_data = data  # same data already loaded for calibration

has_test = 'X_test' in eval_data.files and eval_data['y_test'].sum() > 0
if has_test:
    if args.longctx or args.g_features:
        X_eval = eval_data['X_test'].astype('float32')
    else:
        X_eval = eval_data['X_test'][..., np.newaxis].astype('float32')
    y_eval = eval_data['y_test']
    eval_label = 'test'
else:
    if args.longctx or args.g_features:
        X_eval_train = eval_data['X_train'].astype('float32')
    else:
        X_eval_train = eval_data['X_train'][..., np.newaxis].astype('float32')
    y_eval_train = eval_data['y_train']
    X_eval = X_eval_train[:500]
    y_eval = y_eval_train[:500]
    eval_label = 'train (no test seizures)'

print(f"\nSNN evaluation on {eval_label} set ({len(X_eval)} windows)...")
snn_preds_raw = akida_model.predict(X_eval)

# predict() returns (N, 1, 1, C) — MUST squeeze before argmax
snn_preds = np.argmax(snn_preds_raw.squeeze(), axis=1)

if len(np.unique(y_eval)) == 2:
    cm   = confusion_matrix(y_eval, snn_preds)
    tn, fp, fn, tp = cm.ravel()
    sens   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    n_neg  = tn + fp
    fpr_hr = fp / (n_neg * 2 / 3600) if n_neg > 0 else 0.0
    f1     = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    print(f"  Sensitivity : {sens:.4f}")
    print(f"  Specificity : {spec:.4f}")
    print(f"  FPR / hour  : {fpr_hr:.2f}")
    print(f"  F1          : {f1:.4f}")
else:
    sens = spec = fpr_hr = f1 = None
    print("  (Single class in eval — metrics not applicable)")

# ── 7. Save results ───────────────────────────────────────────────────────────
results = {
    'base_tag'         : base_tag,
    'patient'          : args.patient,
    'eval_patient'     : eval_tag,
    'variant'          : args.variant,
    'model_version'    : args.model_version,
    'weight_bits'      : args.w_bits,
    'activation_bits'  : args.a_bits,
    'cal_samples'      : len(X_cal),
    'eval_set'         : eval_label,
    'n_eval'           : int(len(y_eval)),
    'n_seizure'        : int(y_eval.sum()),
    'sensitivity'      : round(sens, 4) if sens is not None else None,
    'specificity'      : round(spec, 4) if spec is not None else None,
    'fpr_per_hour'     : round(fpr_hr, 2) if fpr_hr is not None else None,
    'f1'               : round(f1, 4) if f1 is not None else None,
}
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {json_path}")
print(f"SNN model: {fbz_path}")
print("\nNext: SCP the .fbz to the Pi and run run_on_akida.py")
