"""
src/evaluation/eval_ann_event_level.py
========================================
Gate 3 — float-ANN equivalent of eval_event_level.py. Same event-level +
collapse + latency metric bundle, but running inference through the Keras
checkpoint directly (argmax classification) rather than the SNN simulator.
A.6 requires every metric reported at BOTH the float-ANN stage and
post-SNN-conversion — the ANN->SNN delta is itself a result.

Usage:
    python3 src/evaluation/eval_ann_event_level.py \
        --ckpt results/best_ann_chb10_v2.h5 --eval-patient chb10

    # cross-patient eval of a pool-trained checkpoint
    python3 src/evaluation/eval_ann_event_level.py \
        --ckpt results/best_ann_multi_v2.h5 --eval-patient chb10
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

import argparse, json, sys
import numpy as np
import tf_keras as keras

sys.path.insert(0, '.')
from src.evaluation.sliding_vote import event_level_metrics, collapse_diagnostic
from src.manifest import load_manifest, require_scaler_match

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, help='Path to a Keras .h5 checkpoint')
parser.add_argument('--eval-patient', required=True)
parser.add_argument('--scaler-source', default=None,
                    help="Same Gate 0b override pattern as eval_event_level.py.")
parser.add_argument('--gap-tolerance', type=float, default=60.0)
parser.add_argument('--detection-fraction', type=float, default=0.3)
parser.add_argument('--min-sustained', type=int, default=3)
args = parser.parse_args()

if not os.path.exists(args.ckpt):
    sys.exit(f"ERROR: checkpoint not found: {args.ckpt}")

data_path = f'data/processed/{args.eval_patient}_dataset_ann.npz'
if not os.path.exists(data_path):
    sys.exit(f"ERROR: {data_path} not found.\n"
             f"Run: python3 src/preprocessing/build_dataset.py --patient {args.eval_patient}")

data = np.load(data_path)
has_test = 'X_test' in data.files and data['y_test'].sum() > 0
if has_test:
    X_eval = data['X_test'][..., np.newaxis].astype('float32')
    y_eval = data['y_test']
else:
    X_eval = data['X_val'][..., np.newaxis].astype('float32')
    y_eval = data['y_val']
    print("WARNING: no test-split seizures — falling back to val split.")

print(f"Model        : {args.ckpt}")
print(f"Eval patient : {args.eval_patient}")
print(f"Eval set     : {len(X_eval)} windows, {int(y_eval.sum())} seizure windows")

# ── Gate 0b-style scaler override ──────────────────────────────────────────
own_scaler_path = f'data/processed/{args.eval_patient}_scaler.json'
with open(own_scaler_path) as f:
    own_scaler = json.load(f)

if args.scaler_source:
    with open(args.scaler_source) as f:
        src = json.load(f)
    if 'per_patient' in src:
        if args.eval_patient not in src['per_patient']:
            sys.exit(f"ERROR: {args.eval_patient} has no entry in "
                      f"{args.scaler_source}'s per_patient map.")
        override_scaler = src['per_patient'][args.eval_patient]
    else:
        override_scaler = src
    print(f"\n[Scaler override] {own_scaler_path} -> {args.scaler_source}")
    X_raw = (X_eval - own_scaler['shift']) / own_scaler['scale']
    X_eval = np.clip(X_raw * override_scaler['scale'] + override_scaler['shift'],
                      0.0, 255.0).astype('float32')
else:
    print(f"\n[Scaler] using {own_scaler_path} as baked into the npz "
          "(no --scaler-source override given)")

# ── Gate 1b-style manifest check (with the "never pooled" fix) ────────────
ckpt_manifest = load_manifest(args.ckpt, required=False)
if ckpt_manifest is None:
    print(f"\n[Gate 1b] WARNING: no manifest for {args.ckpt} — scaler "
          "consistency cannot be verified.")
else:
    model_scaler = ckpt_manifest.get('scaler')
    if model_scaler is None:
        print(f"\n[Gate 1b] WARNING: {args.ckpt}'s manifest has no scaler recorded.")
    else:
        if 'per_patient' in model_scaler:
            if args.eval_patient not in model_scaler['per_patient']:
                print(f"\n[Gate 1b] NOTE: {args.eval_patient} was never one of "
                      f"{args.ckpt}'s pool constituents — no pool-relative scaler "
                      "to check against. Normal generalisation case, not the "
                      "chb06 dual-scaler bug. Proceeding without a scaler check.")
                expected_scaler = None
            else:
                expected_scaler = model_scaler['per_patient'][args.eval_patient]
        else:
            expected_scaler = model_scaler

        if expected_scaler is not None:
            actual_scaler = override_scaler if args.scaler_source else own_scaler
            require_scaler_match(
                expected_scaler, actual_scaler,
                context=f"{args.ckpt} (trained) vs. eval input for "
                         f"{args.eval_patient} (this run)",
            )
            print(f"\n[Gate 1b] Scaler consistency verified against "
                  f"{args.ckpt}'s manifest ✓")

# ── Inference (float ANN — plain argmax, no spike threshold) ──────────────
print(f"\nLoading model and running inference...")
model = keras.models.load_model(args.ckpt)
chunk = 256
y_pred = np.empty(len(X_eval), dtype=np.int32)
for s in range(0, len(X_eval), chunk):
    y_pred[s:s+chunk] = np.argmax(model.predict(X_eval[s:s+chunk], verbose=0), axis=1)

tp = int(((y_pred == 1) & (y_eval == 1)).sum())
fp = int(((y_pred == 1) & (y_eval == 0)).sum())
fn = int(((y_pred == 0) & (y_eval == 1)).sum())
tn = int(((y_pred == 0) & (y_eval == 0)).sum())
win_sens = tp / (tp + fn) if (tp + fn) > 0 else None
win_spec = tn / (tn + fp) if (tn + fp) > 0 else None
n_neg = tn + fp
win_fpr_hr = fp / (n_neg * 2.0 / 3600) if n_neg > 0 else None

print(f"\n=== Window-level ===")
print(f"  Sensitivity : {win_sens:.4f}")
print(f"  Specificity : {win_spec:.4f}")
print(f"  FPR / hour  : {win_fpr_hr:.2f}")

ev = event_level_metrics(
    y_eval, y_pred, window_s=2.0, overlap=0.5,
    detection_fraction=args.detection_fraction,
    min_sustained_windows=args.min_sustained,
    gap_tolerance_s=args.gap_tolerance,
)
print(f"\n=== Event-level (gap={args.gap_tolerance}s, "
      f"frac={args.detection_fraction}, sustained={args.min_sustained}) ===")
print(f"  Event sensitivity : {ev['event_sensitivity']}")
print(f"  Events detected   : {ev['n_detected']}/{ev['n_events']}")
print(f"  False positive events : {ev['false_positives']}")
print(f"  FP events / hour  : {ev['fp_per_hour']}")
print(f"  Total hours       : {ev['total_hours']:.2f}")
print(f"  Detection latency (s) : {ev['latencies_s']}")
print(f"  Mean latency (s)      : {ev['mean_latency_s']}")
print(f"  Undetected events     : {ev['n_undetected']}")

collapse = collapse_diagnostic(
    y_pred, window_specificity=win_spec, y_true=y_eval, step_s=1.0,
    gap_tolerance_s=args.gap_tolerance,
)
print(f"\n=== Collapse diagnostic ===")
print(f"  Positive-window fraction : {collapse['positive_window_fraction']}")
print(f"  Predicted blocks         : {collapse['n_predicted_blocks']}")
print(f"  Largest block fraction   : {collapse['largest_block_fraction']}")
flag = "PASS" if collapse['pass'] else "FAIL"
print(f"  Verdict                  : {flag}")
if not collapse['pass']:
    for r in collapse['reasons']:
        print(f"    - {r}")

out_path = (f"results/ann_event_results_"
            f"{os.path.basename(args.ckpt).replace('.h5','')}"
            f"_on_{args.eval_patient}.json")
with open(out_path, 'w') as f:
    json.dump({
        'ckpt_path': args.ckpt,
        'eval_patient': args.eval_patient,
        'scaler_source': args.scaler_source if args.scaler_source else own_scaler_path,
        'collapse_diagnostic': collapse,
        'window_level': {
            'sensitivity': round(win_sens, 4) if win_sens is not None else None,
            'specificity': round(win_spec, 4) if win_spec is not None else None,
            'fpr_per_hour': round(win_fpr_hr, 2) if win_fpr_hr is not None else None,
        },
        'event_level': {
            'event_sensitivity': ev['event_sensitivity'],
            'n_events': ev['n_events'],
            'n_detected': ev['n_detected'],
            'false_positives': ev['false_positives'],
            'fp_per_hour': ev['fp_per_hour'],
            'total_hours': ev['total_hours'],
            'gap_tolerance_s': args.gap_tolerance,
            'detection_fraction': args.detection_fraction,
            'min_sustained_windows': args.min_sustained,
            'latencies_s': ev['latencies_s'],
            'mean_latency_s': ev['mean_latency_s'],
            'n_undetected': ev['n_undetected'],
        },
    }, f, indent=2)
print(f"\nSaved: {out_path}")
