
"""
src/preprocessing/compute_calib_scaler.py
==========================================
Item 2 (per-patient online amplitude calibration) — Gate scope per
Handoff_gates0_3_multipatient_longctx_complete.md §3.

Simulates a realistic deployment calibration step: collect the first
N seconds of a NEW patient's own EEG (non-seizure only), compute a
min-max scaler from that short window alone, and save it in the same
flat {scale, shift} JSON format eval_event_level.py's --scaler-source
already consumes. No retraining, no architecture change — this only
changes what gets fed into the frozen pool model at inference time.

Scaler convention matches build_dataset.py exactly (min-max to [0,255],
NOT z-score — the z-score path in preprocess.py is commented out and
not the convention in use):
    scale = 255 / (X_max - X_min)
    shift = -X_min * scale

Hard-refuses if the calibration window contains ANY seizure windows —
the whole point is a clean, label-free "first few minutes of signal"
calibration; silently mixing in ictal amplitude would defeat the
deployment realism this is meant to test.

Usage:
    python3 src/preprocessing/compute_calib_scaler.py --patient chb03
    python3 src/preprocessing/compute_calib_scaler.py --patient chb13 --calib-seconds 180
"""
import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--patient', required=True,
                     help="Patient tag, e.g. chb03")
parser.add_argument('--calib-seconds', type=int, default=120,
                     help="Length of the calibration window in seconds "
                          "(default 120 = 2 minutes). Windows are taken "
                          "chronologically from the START of the "
                          "patient's recording, with ~1s stride per the "
                          "project's WINDOW_S=2.0/OVERLAP=0.5 convention.")
parser.add_argument('--input-dir', default='data/processed/')
parser.add_argument('--output', default=None,
                     help="Output scaler JSON path. Default: "
                          "data/processed/{patient}_calib{N}s_scaler.json")
args = parser.parse_args()

pid = args.patient
X_path = os.path.join(args.input_dir, f'{pid}_X.npy')
y_path = os.path.join(args.input_dir, f'{pid}_y.npy')
assert os.path.exists(X_path) and os.path.exists(y_path), \
    f"Raw windows not found: {X_path} / {y_path}\n" \
    f"These are the pre-split, pre-scaling arrays from preprocess.py — " \
    f"same inputs build_dataset.py reads."

X = np.load(X_path)
y = np.load(y_path)
print(f"Patient        : {pid}")
print(f"Full recording : {X.shape[0]} windows "
      f"({int(y.sum())} seizure windows total)")

# Window stride is ~1.0s per the project's fixed WINDOW_S=2.0/OVERLAP=0.5
# convention (step_samples = window_samples * 0.5 => 1s at 256Hz). Window
# index i therefore starts at approximately t=i seconds.
n_calib_windows = args.calib_seconds
n_calib_windows = min(n_calib_windows, X.shape[0])

X_calib = X[:n_calib_windows]
y_calib = y[:n_calib_windows]

n_seizure_in_calib = int(y_calib.sum())
if n_seizure_in_calib > 0:
    raise SystemExit(
        f"REFUSING: first {args.calib_seconds}s of {pid}'s recording "
        f"contains {n_seizure_in_calib} seizure window(s). A calibration "
        f"window must be seizure-free by construction (the deployment "
        f"scenario this simulates is 'plug in the device, record a few "
        f"clean minutes, calibrate'). Either increase/decrease "
        f"--calib-seconds to dodge the early seizure, or pick a "
        f"different start offset manually — do not silently include "
        f"ictal amplitude in the scaler."
    )

print(f"Calibration window : first {args.calib_seconds}s "
      f"({n_calib_windows} windows), confirmed seizure-free")

X_min = float(X_calib.min())
X_max = float(X_calib.max())
assert X_max > X_min, f"Degenerate calibration window for {pid}: " \
                       f"X_min == X_max == {X_min}"

scale = 255.0 / (X_max - X_min)
shift = -X_min * scale

scaler = {
    'scale': scale,
    'shift': shift,
    'X_min': X_min,
    'X_max': X_max,
    'patient': pid,
    'calib_seconds': args.calib_seconds,
    'n_calib_windows': n_calib_windows,
    'source': f'first {args.calib_seconds}s of own recording, non-seizure-only, '
              f'NOT full-recording scaler (cf. {pid}_scaler.json)',
}

out_path = args.output or os.path.join(
    args.input_dir, f'{pid}_calib{args.calib_seconds}s_scaler.json')
with open(out_path, 'w') as f:
    json.dump(scaler, f, indent=2)

# Print alongside the patient's own full-recording scaler for a quick
# sanity comparison (large divergence here is itself informative).
own_scaler_path = os.path.join(args.input_dir, f'{pid}_scaler.json')
print(f"\nCalibration scaler : scale={scale:.4f}  shift={shift:.4f}  "
      f"range=[{X_min:.6f}, {X_max:.6f}]")
if os.path.exists(own_scaler_path):
    with open(own_scaler_path) as f:
        own = json.load(f)
    print(f"Full-recording scaler (reference) : scale={own['scale']:.4f}  "
          f"shift={own['shift']:.4f}  "
          f"range=[{own['X_min']:.6f}, {own['X_max']:.6f}]")
    pct_diff = 100 * (scale - own['scale']) / own['scale']
    print(f"Scale divergence vs full-recording scaler : {pct_diff:+.1f}%")

print(f"\nSaved: {out_path}")
print(f"\nNext: python3 src/evaluation/eval_event_level.py \\\n"
      f"    --fbz results/seizure_model_multi_v2_w4a4.fbz \\\n"
      f"    --eval-patient {pid} \\\n"
      f"    --scaler-source {out_path}")
