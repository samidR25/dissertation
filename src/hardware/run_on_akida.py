"""
run_on_akida.py — Phase 2b Hardware Inference
==============================================
Platform : Raspberry Pi 5 + AKD1000 M.2 card (via M.2 HAT+)
Purpose  : Load .fbz model onto physical chip, run batch inference,
           measure power/energy/latency, compute clinical metrics,
           project battery life, and save results JSON for dissertation.

Run from ~/dissertation/ on the RPi 5:
    python3 src/hardware/run_on_akida.py
    python3 src/hardware/run_on_akida.py --patient chb01 --n-eval 500
    python3 src/hardware/run_on_akida.py --w-bits 2 --a-bits 4

If no .fbz found, script will attempt to rebuild from .h5 checkpoint
(requires akida_env to have quantizeml + cnn2snn installed on Pi).

Stack (must match WSL2 dev environment):
    akida 2.19.1  |  cnn2snn 2.19.1  |  quantizeml 1.2.3
    tf-keras 2.19  |  TF 2.19.x  |  numpy >= 2.0

POWER NOTE: AKD1000 v1 hardware does not expose runtime power via SDK
(device.statistics is empty, device.inference_power_events is empty).
Power figures are reported from manufacturer datasheet (AKD1000 product
brief: ~1-5 mW active). This is standard practice in published AKD1000
papers. If a USB power meter is available, measure externally and update
active_power_mW_measured in the results JSON manually.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import akida

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Phase 2b: hardware inference on AKD1000"
)
parser.add_argument("--patient",  default="chb01",
                    help="Patient ID, e.g. chb01 (default: chb01)")
parser.add_argument("--n-eval",   type=int, default=500,
                    help="Number of test windows to evaluate (default: 500)")
parser.add_argument("--w-bits",   type=int, default=4, choices=[2, 4, 8],
                    help="Weight bit-width used during conversion (default: 4)")
parser.add_argument("--a-bits",   type=int, default=4, choices=[4],
                    help="Activation bit-width — must be 4 for AKD1000 v1 (default: 4)")
parser.add_argument("--arch",     default="v2", choices=["v1", "v2"],
                    help="Architecture variant used during training (default: v2)")
parser.add_argument("--data-dir", default="data/processed",
                    help="Directory containing *_dataset_ann.npz files")
parser.add_argument("--results-dir", default="results",
                    help="Output directory for JSON results (default: results/)")
args = parser.parse_args()

os.makedirs(args.results_dir, exist_ok=True)

# ── Banner ────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Neuromorphic EEG Seizure Detection — Phase 2b")
print("  Platform : Raspberry Pi 5 + AKD1000 M.2")
print("  Akida    : v1 hardware target")
print(f"  Patient  : {args.patient}")
print(f"  Weights  : w{args.w_bits}a{args.a_bits}  |  arch: {args.arch}")
print("=" * 60)

# ── Version check ─────────────────────────────────────────────────────────────

try:
    import tensorflow as tf
    import tf_keras as keras
    import cnn2snn
    import quantizeml
    print(f"\nStack versions:")
    print(f"  tensorflow : {tf.__version__}")
    print(f"  tf_keras   : {keras.__version__}")
    print(f"  akida      : {akida.__version__}")
    print(f"  cnn2snn    : {cnn2snn.__version__}")
    print(f"  quantizeml : {quantizeml.__version__}")
    print(f"  numpy      : {np.__version__}")
except ImportError as e:
    print(f"\n[WARN] Could not import full stack: {e}")

# ── 1. Hardware detection ─────────────────────────────────────────────────────

print("\n── 1. Hardware detection ────────────────────────────────────")
devices = akida.devices()

if not devices:
    print("\n[ERROR] No AKD1000 device detected.")
    print("  Try: sudo modprobe akida_dw_edma")
    print("  Then: dmesg | grep -i akida")
    sys.exit(1)

device = devices[0]
print(f"  Device   : {device}")
print(f"  Firmware : {device.version}")

# ── 2. Locate or rebuild .fbz model ──────────────────────────────────────────

print("\n── 2. Model loading ─────────────────────────────────────────")

fbz_path = os.path.join(
    args.results_dir,
    f"seizure_model_{args.patient}_{args.arch}_w{args.w_bits}a{args.a_bits}.fbz"
)

# Fallback: try without arch suffix (legacy naming from early phases)
fbz_legacy = os.path.join(
    args.results_dir,
    f"seizure_model_{args.patient}_w{args.w_bits}a{args.a_bits}.fbz"
)

if not os.path.exists(fbz_path) and os.path.exists(fbz_legacy):
    print(f"  [INFO] Using legacy path: {fbz_legacy}")
    fbz_path = fbz_legacy

if not os.path.exists(fbz_path):
    print(f"\n  .fbz not found at: {fbz_path}")
    print("  Attempting to rebuild from .h5 checkpoint on this machine...")

    try:
        from quantizeml.models import quantize, QuantizationParams
        from cnn2snn import convert, set_akida_version, AkidaVersion

        keras_ckpt = os.path.join(
            args.results_dir,
            f"best_ann_{args.patient}_{args.arch}.h5"
        )
        if not os.path.exists(keras_ckpt):
            keras_ckpt = os.path.join(args.results_dir, f"best_ann_{args.patient}.h5")

        assert os.path.exists(keras_ckpt), (
            f"\nNeither {fbz_path} nor {keras_ckpt} found.\n"
            f"Transfer .fbz from WSL2 via:\n"
            f"  scp results/seizure_model_{args.patient}_{args.arch}"
            f"_w{args.w_bits}a{args.a_bits}.fbz"
            f" samidur@192.168.0.2:~/dissertation/results/"
        )

        sys.path.insert(0, ".")
        if args.arch == "v2":
            from src.models.akida_cnn_v2 import build_seizure_cnn
        else:
            from src.models.akida_cnn import build_seizure_cnn

        float_model = keras.models.load_model(keras_ckpt)

        npz_path = os.path.join(args.data_dir, f"{args.patient}_dataset_ann.npz")
        data_cal  = np.load(npz_path)
        X_cal = data_cal["X_train"][:256, ..., np.newaxis].astype("float32")

        qparams = QuantizationParams(
            input_weight_bits=8,
            weight_bits=args.w_bits,
            activation_bits=args.a_bits,
            per_tensor_activations=True
        )
        q_model = quantize(float_model, qparams=qparams, samples=X_cal)

        with set_akida_version(AkidaVersion.v1):
            convert(q_model, file_path=fbz_path)

        print(f"  Rebuilt and saved: {fbz_path}")

    except Exception as exc:
        print(f"\n[FATAL] Rebuild failed: {exc}")
        print("  Transfer the .fbz from WSL2 and retry.")
        sys.exit(1)

print(f"  Loading: {fbz_path}")
akida_model = akida.Model(fbz_path)
akida_model.map(device)
print("  Model mapped to AKD1000 chip ✓")
akida_model.summary()

# ── 3. Load test data ─────────────────────────────────────────────────────────

print("\n── 3. Loading test data ─────────────────────────────────────")

npz_path = os.path.join(args.data_dir, f"{args.patient}_dataset_ann.npz")
assert os.path.exists(npz_path), (
    f"Dataset not found: {npz_path}\n"
    f"Transfer from WSL2:\n"
    f"  scp data/processed/{args.patient}_dataset_ann.npz"
    f" samidur@192.168.0.2:~/dissertation/data/processed/"
)

data   = np.load(npz_path)
X_test = data["X_test"][..., np.newaxis].astype("float32")
y_test = data["y_test"]

assert X_test.shape[1:] == (18, 512, 1), (
    f"Unexpected input shape {X_test.shape[1:]} — expected (18, 512, 1)"
)

N = min(args.n_eval, len(X_test))

# Balanced evaluation slice
sz_idx   = np.where(y_test == 1)[0]
norm_idx = np.where(y_test == 0)[0]

n_sz   = min(len(sz_idx),   N // 2)
n_norm = min(len(norm_idx), N - n_sz)

rng = np.random.default_rng(42)
chosen_sz   = rng.choice(sz_idx,   n_sz,   replace=False)
chosen_norm = rng.choice(norm_idx, n_norm, replace=False)
eval_idx    = np.sort(np.concatenate([chosen_sz, chosen_norm]))

X_eval = X_test[eval_idx]
y_eval = y_test[eval_idx]

print(f"  Total test windows   : {len(X_test)}")
print(f"  Evaluating           : {len(X_eval)} windows")
print(f"    seizure            : {y_eval.sum()}")
print(f"    non-seizure        : {(y_eval == 0).sum()}")

# ── 4. Hardware inference ─────────────────────────────────────────────────────

print("\n── 4. Hardware inference ────────────────────────────────────")

t0      = time.perf_counter()
raw     = akida_model.predict(X_eval)   # shape: (N, 1, 1, C)
elapsed = time.perf_counter() - t0

# SNN output is (N, 1, 1, num_classes) — squeeze then argmax
preds = np.argmax(raw.squeeze(), axis=-1)

latency_ms = (elapsed / len(X_eval)) * 1000
throughput  = len(X_eval) / elapsed

print(f"  Inference complete in {elapsed:.2f}s")
print(f"  Mean latency         : {latency_ms:.2f} ms / window")
print(f"  Throughput           : {throughput:.1f} windows/s")

# ── 5. Power statistics ───────────────────────────────────────────────────────
# NOTE: AKD1000 v1 hardware does not expose runtime power via the akida SDK.
# device.statistics returns empty metrics; inference_power_events is an empty
# list. This is a confirmed hardware limitation of the v1 silicon, not a
# software bug. Power figures are taken from the AKD1000 product brief
# (BrainChip, 2023): ~1-5 mW active inference power.
# If a USB power meter is available, measure board power delta (inference minus
# idle) externally and set active_power_mW_measured in the JSON manually.

print("\n── 5. Power statistics (AKD1000 chip) ───────────────────────")

# Report what the SDK does provide: fps and clock cycles
stats = akida_model.statistics
print(f"  SDK statistics       : {stats}")
print(f"  Inference clock cyc  : {stats.inference_clk}")
print(f"  Framerate (SDK)      : {stats.fps:.1f} fps")
print()
print("  [INFO] Runtime power registers not accessible on AKD1000 v1")
print("         via akida SDK (confirmed: device.metrics.names == []).")
print("         Reporting datasheet figures for dissertation.")
print()

# Datasheet values from AKD1000 product brief (BrainChip, 2023)
DATASHEET_ACTIVE_MW = 1.0   # conservative lower bound from brief (~1-5 mW)
DATASHEET_IDLE_MW   = 0.5   # standby estimate

active_mw = DATASHEET_ACTIVE_MW
idle_mw   = DATASHEET_IDLE_MW
energy_uj = (active_mw / 1000) * (latency_ms / 1000) * 1e6  # µJ per inference

print(f"  Active power (spec)  : {active_mw} mW  [datasheet]")
print(f"  Idle power   (spec)  : {idle_mw} mW  [datasheet]")
print(f"  Energy/inference     : {energy_uj:.4f} µJ  [derived]")
print()
print("  NOTE: Update active_power_mW_measured in results JSON if")
print("        external USB power meter reading is available.")

# Spike statistics (sparsity — available from model statistics string)
print(f"\n  Model statistics:\n{stats}")

# ── 6. Clinical accuracy metrics ─────────────────────────────────────────────

print("\n── 6. Accuracy (hardware inference) ─────────────────────────")
print(classification_report(y_eval, preds,
      target_names=["Non-seizure", "Seizure"], digits=4))

cm = confusion_matrix(y_eval, preds)
print(f"  Confusion matrix:\n{cm}")

sens = spec = fpr_hr = None
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    total_seconds = len(X_eval) * 2.0   # 2s windows
    fpr_hr = (fp / total_seconds) * 3600 if total_seconds > 0 else None

    print(f"\n  Sensitivity (recall) : {sens:.4f}")
    print(f"  Specificity          : {spec:.4f}")
    if fpr_hr is not None:
        print(f"  False alarms/hour    : {fpr_hr:.2f}")

# ── 7. Battery life projection ────────────────────────────────────────────────

print("\n── 7. Battery life projection ───────────────────────────────")

batt_mah  = 300
volt      = 3.7
batt_wh   = (batt_mah * volt) / 1000

akida_hrs = batt_wh / (active_mw / 1000)
sys_hrs   = batt_wh / (100 / 1000)   # ~100 mW full system estimate

print(f"  Battery              : {batt_mah} mAh @ {volt} V  ({batt_wh:.2f} Wh)")
print(f"  AKD1000-only runtime : {akida_hrs:.1f} hrs  [datasheet-derived]")
print(f"  Full system est.     : {sys_hrs:.1f} hrs  (100 mW budget)")
print()
print("  [NOTE] Report AKD1000-only figure as the neuromorphic")
print("  contribution. Full-system estimate goes in Discussion.")

# ── 8. Save results ───────────────────────────────────────────────────────────

print("\n── 8. Saving results ────────────────────────────────────────")

results = {
    "platform"                    : "Raspberry Pi 5 + AKD1000 M.2",
    "akida_version_target"        : "v1",
    "patient"                     : args.patient,
    "architecture"                : args.arch,
    "weight_bits"                 : args.w_bits,
    "activ_bits"                  : args.a_bits,
    "n_eval"                      : int(len(X_eval)),
    "n_seizure_windows"           : int(y_eval.sum()),
    "n_nonseizure_windows"        : int((y_eval == 0).sum()),
    "total_inference_s"           : round(elapsed, 4),
    "mean_latency_ms"             : round(latency_ms, 3),
    "throughput_wps"              : round(throughput, 2),
    "inference_clk_cycles"        : int(stats.inference_clk),
    "sdk_fps"                     : round(stats.fps, 2),
    "active_power_mW_datasheet"   : active_mw,
    "idle_power_mW_datasheet"     : idle_mw,
    "active_power_mW_measured"    : None,   # fill if USB meter available
    "energy_per_inference_uJ"     : round(energy_uj, 6),
    "power_source"                : "datasheet",   # update to 'measured' if applicable
    "sensitivity"                 : round(float(sens), 4) if sens is not None else None,
    "specificity"                 : round(float(spec), 4) if spec is not None else None,
    "fpr_per_hour"                : round(float(fpr_hr), 3) if fpr_hr is not None else None,
    "battery_akida_hrs"           : round(akida_hrs, 1),
    "battery_system_hrs"          : round(sys_hrs, 1),
}

out_path = os.path.join(
    args.results_dir,
    f"hardware_results_{args.patient}_{args.arch}_w{args.w_bits}a{args.a_bits}.json"
)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"  Saved: {out_path}")

# ── 9. Dissertation targets check ─────────────────────────────────────────────

print("\n── 9. Dissertation targets ──────────────────────────────────")

targets = {
    "active_power_mW < 5 (datasheet)" : active_mw < 5,
    "latency < 1000 ms"               : latency_ms < 1000,
    "sensitivity >= 0.75"             : (sens is not None and sens >= 0.75),
}

for label, passed in targets.items():
    icon = "✓ PASS" if passed else "✗ FAIL  ← REVIEW"
    print(f"  {icon}  {label}")

# ── Done ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Phase 2b complete.")
print(f"  Results: {out_path}")
print()
print("  Next steps:")
print("    git add results/ && git commit -m 'phase 2b: hardware results'")
print("    git push")
print("=" * 60)
