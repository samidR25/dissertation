"""
src/hardware/run_on_akida.py
=============================
Hardware inference on physical AKD1000 chip.
Phase 2b/2c — updated threshold sweep to cover 0.2–0.8 bidirectionally.

Key change vs previous version:
  --calibrate-threshold now sweeps [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8]
  This lets us find operating points BELOW 0.5 (recover sensitivity from Phase 2b model)
  as well as above 0.5 (reduce FPR).

Usage:
    # Calibration sweep (Phase 2b model, recovering sensitivity/FPR tradeoff)
    python3 src/hardware/run_on_akida.py --patient chb03 --model-version 2 \
        --n-eval 500 --calibrate-threshold --smooth-k 5

    # Run chosen operating point
    python3 src/hardware/run_on_akida.py --patient chb03 --model-version 2 \
        --n-eval 500 --spike-threshold 0.35 --smooth-k 5

    # Phase 2c base model
    python3 src/hardware/run_on_akida.py --patient chb03 --model-version 2 \
        --base multi --n-eval 500 --spike-threshold 0.35 --smooth-k 5
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
import argparse, json, os, sys, time
import numpy as np
import tf_keras as keras
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import convert, set_akida_version, AkidaVersion
import akida

parser = argparse.ArgumentParser()
parser.add_argument('--patient',             default='chb01')
parser.add_argument('--model-version',       type=int,   default=2)
parser.add_argument('--base',                default=None,
                    help="Override fbz base tag. E.g. 'multi'.")
parser.add_argument('--n-eval',              type=int,   default=500)
parser.add_argument('--w-bits',              type=int,   default=4)
parser.add_argument('--a-bits',              type=int,   default=4)
parser.add_argument('--smooth-k',            type=int,   default=0,
                    help="Majority vote k (0=disabled).")
parser.add_argument('--spike-threshold',     type=float, default=0.5,
                    help="Seizure spike ratio threshold (0.5=argmax).")
parser.add_argument('--calibrate-threshold', action='store_true',
                    help="Sweep thresholds 0.2–0.8 and print table. No JSON saved.")
args = parser.parse_args()

os.makedirs('results', exist_ok=True)

# ── 1. Device ─────────────────────────────────────────────────────────────────
print("=== AKD1000 Hardware Detection ===")
devices = akida.devices()
if not devices:
    raise RuntimeError(
        "No AKD1000 device found.\n"
        "  sudo modprobe akida_dw_edma\n"
        "  lspci | grep -i brain")
device = devices[0]
print(f"Device  : {device}")
print(f"Firmware: {device.version}")

# ── 2. Model path ─────────────────────────────────────────────────────────────
model_tag = args.base if args.base else args.patient
fbz_path  = (f'results/seizure_model_{model_tag}'
             f'_v{args.model_version}_w{args.w_bits}a{args.a_bits}.fbz')

if not os.path.exists(fbz_path):
    sys.exit(f"ERROR: {fbz_path} not found. Run convert_to_snn.py first.")

print(f"\nLoading: {fbz_path}")
akida_model = akida.Model(fbz_path)
akida_model.map(device)
print("Model mapped to AKD1000 ✓")

# ── 3. Load test data (balanced eval set) ─────────────────────────────────────
data_path = f'data/processed/{args.patient}_dataset_ann.npz'
data      = np.load(data_path)
X_test_full = data['X_test'][..., np.newaxis].astype('float32')
y_test_full = data['y_test']

seiz_idx = np.where(y_test_full == 1)[0]
nons_idx = np.where(y_test_full == 0)[0]
n_seiz   = len(seiz_idx)

if n_seiz > 0:
    half     = args.n_eval // 2
    s_idx    = seiz_idx[:half] if len(seiz_idx) >= half else seiz_idx
    n_idx    = nons_idx[:half] if len(nons_idx) >= half else nons_idx
    eval_idx = np.sort(np.concatenate([s_idx, n_idx]))
else:
    eval_idx = np.arange(min(args.n_eval, len(y_test_full)))

X_eval = X_test_full[eval_idx]
y_eval = y_test_full[eval_idx]
N      = len(X_eval)
print(f"\nEval set: {N} windows "
      f"(seizure={int(y_eval.sum())}, non-sz={int((y_eval==0).sum())})")

# ── 4. Hardware inference ─────────────────────────────────────────────────────
print("\nRunning inference on AKD1000...")
t0        = time.perf_counter()
preds_raw = akida_model.predict(X_eval)
elapsed   = time.perf_counter() - t0
latency_ms = (elapsed / N) * 1000
print(f"Done: {elapsed:.2f}s  ({latency_ms:.2f} ms/window)")

# AKD1000 v1: (N,1,1,C) → squeeze → (N,2)
spike_counts = preds_raw.squeeze()

stats = akida_model.statistics
print(f"\nSDK stats: fps={stats.fps:.1f}  clk={stats.inference_clk}")

# ── 5. Helper ─────────────────────────────────────────────────────────────────
def _apply(spike_counts, threshold, smooth_k):
    # Sign-safe sigmoid-of-margin (matches eval_event_level.py's fix) --
    # the old ratio=count1/total formula silently inverts the decision
    # whenever total<0, since spike_counts are signed potentials, not
    # guaranteed non-negative spike counts.
    margin = spike_counts[:, 1] - spike_counts[:, 0]
    ratio = 1.0 / (1.0 + np.exp(-margin))
    raw = (ratio >= threshold).astype(np.int32)
    if smooth_k > 1:
        from src.evaluation.sliding_vote import sliding_majority_vote
        return sliding_majority_vote(raw, k=smooth_k), raw
    return raw, raw

def _metrics(preds, y_true, window_sec=2.0):
    tp = int(((preds==1)&(y_true==1)).sum())
    fp = int(((preds==1)&(y_true==0)).sum())
    fn = int(((preds==0)&(y_true==1)).sum())
    tn = int(((preds==0)&(y_true==0)).sum())
    sens   = tp/(tp+fn) if (tp+fn)>0 else None
    spec   = tn/(tn+fp) if (tn+fp)>0 else None
    n_neg  = tn+fp
    fpr_hr = fp/(n_neg*window_sec/3600) if n_neg>0 else None
    return sens, spec, fpr_hr, tp, fp, fn, tn

# ── 6. Calibration sweep ──────────────────────────────────────────────────────
if args.calibrate_threshold:
    # Extended sweep: below 0.5 recovers sensitivity, above 0.5 reduces FPR
    thresholds = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75, 0.80]
    k_vals     = [0]
    if args.smooth_k > 0:
        k_vals.append(args.smooth_k)

    for k in k_vals:
        label = f"vote k={k}" if k > 0 else "no vote"
        print(f"\n=== Threshold Calibration [{label}] ===")
        print(f"  {'Thresh':>7}  {'Sensitivity':>11}  {'Specificity':>11}  "
              f"{'FPR/hr':>8}  {'TP':>4}  {'FP':>5}")
        print("  " + "─" * 52)
        for thr in thresholds:
            p, _ = _apply(spike_counts, thr, k)
            s, sp, fpr, tp, fp, fn, tn = _metrics(p, y_eval)
            s_s   = f"{s:.4f}"   if s   is not None else "  N/A "
            sp_s  = f"{sp:.4f}"  if sp  is not None else "  N/A "
            fpr_s = f"{fpr:.1f}" if fpr is not None else "  N/A"
            print(f"  {thr:>7.2f}  {s_s:>11}  {sp_s:>11}  "
                  f"{fpr_s:>8}  {tp:>4}  {fp:>5}")

    print("\n── How to read this table ──────────────────────────────────────────")
    print("  Below 0.50: higher sensitivity, higher FPR (Phase 2b model direction)")
    print("  Above 0.50: lower FPR, lower sensitivity  (Phase 2c direction)")
    print("  With vote:  FPR drops, sensitivity drops slightly")
    print("  Target: sensitivity >= 0.40 AND FPR <= 100/hr")
    print("  Re-run with: --spike-threshold <chosen> [--smooth-k 5]")
    sys.exit(0)

# ── 7. Primary evaluation ─────────────────────────────────────────────────────
ACTIVE_MW = 1.0
energy_uj = (ACTIVE_MW/1000) * (latency_ms/1000) * 1e6

print(f"\n=== Primary Evaluation ===")
print(f"  Spike threshold : {args.spike_threshold}")
print(f"  Majority vote k : {args.smooth_k if args.smooth_k>0 else 'disabled'}")

final_preds, _ = _apply(spike_counts, args.spike_threshold, args.smooth_k)
sens, spec, fpr_hr, tp, fp, fn, tn = _metrics(final_preds, y_eval)

print(f"\n  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
if sens  is not None: print(f"  Sensitivity : {sens:.4f}")
if spec  is not None: print(f"  Specificity : {spec:.4f}")
if fpr_hr is not None: print(f"  FPR / hour  : {fpr_hr:.2f}")
print(f"  Latency     : {latency_ms:.2f} ms/window")
print(f"  Energy      : {energy_uj:.4f} µJ/inference")

# ── 8. Save ───────────────────────────────────────────────────────────────────
suffix_parts = []
if args.spike_threshold != 0.5:
    suffix_parts.append(f'thr{int(args.spike_threshold*100)}')
if args.smooth_k > 0:
    suffix_parts.append(f'k{args.smooth_k}')
if args.base:
    suffix_parts.append(f'base-{args.base}')
suffix = ('_' + '_'.join(suffix_parts)) if suffix_parts else ''

out_path = (f'results/hardware_results_{args.patient}'
            f'_v{args.model_version}_w{args.w_bits}a{args.a_bits}{suffix}.json')

results = {
    'platform'              : 'Raspberry Pi 5 + AKD1000 M.2',
    'patient'               : args.patient,
    'model_tag'             : model_tag,
    'model_version'         : args.model_version,
    'weight_bits'           : args.w_bits,
    'activ_bits'            : args.a_bits,
    'spike_threshold'       : args.spike_threshold,
    'smooth_k'              : args.smooth_k,
    'n_eval'                : N,
    'n_seizure_windows'     : int(y_eval.sum()),
    'mean_latency_ms'       : round(latency_ms, 3),
    'throughput_wps'        : round(N/elapsed, 1),
    'energy_per_inf_uJ'     : round(energy_uj, 4),
    'active_power_mW'       : ACTIVE_MW,
    'inference_clk_cycles'  : int(stats.inference_clk),
    'sdk_fps'               : round(float(stats.fps), 1),
    'sensitivity'           : round(sens, 4) if sens is not None else None,
    'specificity'           : round(spec, 4) if spec is not None else None,
    'fpr_per_hour'          : round(fpr_hr, 2) if fpr_hr is not None else None,
    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
}
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
