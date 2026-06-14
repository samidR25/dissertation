"""
build_dataset_stft.py — Phase 2d: 3-channel (raw EEG + delta + beta) dataset builder.

MEMORY DESIGN — why this approach
----------------------------------
Problem: chb03 has 128,247 raw windows. A full (N, 18, 512, 3) float32 array
is 13.2 GB. WSL2 has 12 GB RAM and the disk has only 7.4 GB free — neither
approach (hold in RAM, write to temp file) fits.

Solution: don't process from raw windows at all.
The existing chbXX_dataset_ann.npz already contains the split and SMOTE'd data:
  X_train: ~8,980 windows   (SMOTE balanced)   →  0.31 GB → 0.92 GB as 3ch
  X_val:   ~19,237 windows  (real distribution) →  0.66 GB → 1.98 GB as 3ch
  X_test:  ~19,238 windows  (real distribution) →  0.66 GB → 1.98 GB as 3ch
  Total 3-channel output: ~4.9 GB — fits comfortably in 12 GB RAM.

The SMOTE'd X_train already has the right class balance.
The val/test splits already reflect the real (imbalanced) distribution.
All chronological integrity is preserved — those splits were built chronologically.

The frequency features (delta/beta envelopes) are added ON TOP of the already-split
data. The scaler is fitted on X_train (SMOTE'd) and applied to val/test — identical
to what a from-scratch build would do, just starting from smaller arrays.

The three channels:
  ch0: Raw EEG          — from X_train/val/test directly (already [0,255])
  ch1: Delta power map  — 0.5–4 Hz bandpass of ch0, RMS envelope, rescaled
  ch2: Beta power map   — 13–30 Hz bandpass of ch0, RMS envelope, rescaled

Important: ch0 is already scaled to [0,255] in chbXX_dataset_ann.npz.
The bandpass filters are applied to the [0,255]-scaled data. This is fine —
bandpass filtering is a linear operation and the band power ratios are preserved
under linear scaling. The resulting envelopes are then min-max scaled
independently to [0,255] via the per-channel scaler.

Usage
-----
  python3 src/preprocessing/build_dataset_stft.py --patient chb03
  python3 src/preprocessing/build_dataset_stft.py --multi-patient

Gate check:
  python3 -c "
  import numpy as np
  d = np.load('data/processed/chb03_dataset_stft.npz')
  print('X_train:', d['X_train'].shape)   # (N, 18, 512, 3)
  print('range:', d['X_train'].min(), d['X_train'].max())   # [0, 255]
  print('test seizures:', d['y_test'].sum())   # 167
  assert d['X_train'].shape[1:] == (18, 512, 3)
  print('Gate PASSED')
  "

Output files
------------
  data/processed/chbXX_dataset_stft.npz     keys: X_train/y_train/X_val/y_val/X_test/y_test
  data/processed/multi_dataset_stft.npz     keys: X_train/y_train/X_val/y_val  (no test)
  data/processed/chbXX_scaler_stft.json     per-channel scale params
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.ndimage import uniform_filter1d

# ── Constants ──────────────────────────────────────────────────────────────────
SFREQ            = 256
N_SAMPLES        = 512
N_CHANNELS       = 18
DELTA_BAND       = (0.5, 4.0)
BETA_BAND        = (13.0, 30.0)
RMS_WIN_SAMPLES  = 64
MULTI_PATIENTS   = ['chb01', 'chb02', 'chb03', 'chb05']

# Chunk size for the filter step only.
# Each chunk: 3 × chunk × 18 × 512 × 4 bytes peak.
# chunk=5000 → 3 × 5000 × 18 × 512 × 4 = 552 MB peak. Very safe.
CHUNK_SIZE = 5000


# ── Filter helpers ─────────────────────────────────────────────────────────────

def butter_bandpass_sos(lowcut, highcut, fs=SFREQ, order=4):
    nyq  = 0.5 * fs
    low  = np.clip(lowcut  / nyq, 1e-6, 1.0 - 1e-6)
    high = np.clip(highcut / nyq, 1e-6, 1.0 - 1e-6)
    return butter(order, [low, high], btype='band', output='sos')


def compute_band_envelope(X: np.ndarray, band: tuple, label: str = '') -> np.ndarray:
    """
    Compute RMS power envelope for one frequency band.

    Args:
        X:     (N, 18, 512) float32 — input windows (any amplitude range)
        band:  (lowcut_hz, highcut_hz)
        label: progress label string

    Returns:
        (N, 18, 512) float32 — RMS envelope, non-negative

    Memory per chunk (chunk=5000): bandpass output + sq + rms ≈ 552 MB.
    After del, all intermediates are freed before the next chunk.
    """
    N = len(X)
    sos = butter_bandpass_sos(*band)
    out = np.empty_like(X)  # (N, 18, 512) — same size as input, fits in RAM

    n_chunks = (N + CHUNK_SIZE - 1) // CHUNK_SIZE
    t0 = time.time()

    for ci in range(n_chunks):
        s = ci * CHUNK_SIZE
        e = min(s + CHUNK_SIZE, N)

        filt = sosfilt(sos, X[s:e], axis=-1)          # (chunk, 18, 512)
        sq   = filt ** 2
        rms  = uniform_filter1d(sq, size=RMS_WIN_SAMPLES, axis=-1, mode='reflect')
        np.sqrt(np.maximum(rms, 0.0), out=rms)         # in-place sqrt
        out[s:e] = rms
        del filt, sq, rms

        elapsed = time.time() - t0
        rate    = e / elapsed
        remain  = (N - e) / rate if rate > 0 else 0
        pct     = 100.0 * e / N
        tag     = f"[{label}] " if label else ""
        print(f"    {tag}{band[0]:.0f}–{band[1]:.0f}Hz  "
              f"chunk {ci+1}/{n_chunks}  {e}/{N} ({pct:.0f}%)  "
              f"{rate:.0f} win/s  ETA {remain/60:.1f}min",
              flush=True)

    return out


def add_frequency_channels(X_raw: np.ndarray, label: str = '') -> np.ndarray:
    """
    Add delta and beta envelope channels to raw EEG windows.

    Args:
        X_raw: (N, 18, 512) float32 — raw EEG (any scaling, e.g. [0,255])

    Returns:
        (N, 18, 512, 3) float32 — ch0=raw, ch1=delta_env, ch2=beta_env
        (ch1 and ch2 are NOT yet scaled to [0,255] — done separately)
    """
    N = len(X_raw)
    print(f"  Delta envelope ({DELTA_BAND[0]}–{DELTA_BAND[1]} Hz)...")
    delta_env = compute_band_envelope(X_raw, DELTA_BAND, label=label)

    print(f"  Beta envelope ({BETA_BAND[0]}–{BETA_BAND[1]} Hz)...")
    beta_env  = compute_band_envelope(X_raw, BETA_BAND,  label=label)

    # Stack along new last axis — (N, 18, 512, 3)
    out = np.stack([X_raw, delta_env, beta_env], axis=-1).astype(np.float32)
    del delta_env, beta_env
    return out


# ── Per-channel scaler ─────────────────────────────────────────────────────────

def fit_scaler(X_3ch: np.ndarray) -> dict:
    """Fit per-channel min-max scaler on X_3ch (any split). Returns dict."""
    scaler = {}
    for ch in range(3):
        data = X_3ch[..., ch]
        scaler[f'ch{ch}_min'] = float(data.min())
        scaler[f'ch{ch}_max'] = float(data.max())
    return scaler


def apply_scaler(X_3ch: np.ndarray, scaler: dict) -> np.ndarray:
    """Scale each channel independently to [0, 255] float32."""
    out = X_3ch.copy()
    for ch in range(3):
        vmin = scaler[f'ch{ch}_min']
        vmax = scaler[f'ch{ch}_max']
        eps  = 1e-8
        out[..., ch] = np.clip(
            (out[..., ch] - vmin) / (vmax - vmin + eps) * 255.0,
            0.0, 255.0
        ).astype(np.float32)
    return out


# ── Single-patient pipeline ────────────────────────────────────────────────────

def build_single_patient(patient: str, data_dir: str = 'data/processed/') -> None:
    """
    Build chbXX_dataset_stft.npz from chbXX_dataset_ann.npz.

    Loads the already-split, already-SMOTE'd dataset and adds frequency channels.
    Peak RAM: largest split × 18 × 512 × 3 × 4 bytes.
    For chb03: val/test ≈ 19k windows each → ~2 GB each. Well within 12 GB.
    """
    ann_path = os.path.join(data_dir, f'{patient}_dataset_ann.npz')
    if not os.path.exists(ann_path):
        print(f"ERROR: {ann_path} not found.")
        print(f"  Run: python3 src/preprocessing/build_dataset.py --patient {patient}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Building STFT dataset for {patient}")
    print(f"Source: {ann_path}")
    print(f"{'='*60}")

    data = np.load(ann_path)
    X_tr  = data['X_train'].astype(np.float32)
    y_tr  = data['y_train']
    X_val = data['X_val'].astype(np.float32)
    y_val = data['y_val']
    X_te  = data['X_test'].astype(np.float32)
    y_te  = data['y_test']

    # Memory forecast
    def gb(arr): return arr.nbytes / 1024**3
    print(f"\nLoaded splits:")
    print(f"  X_train: {X_tr.shape}   {gb(X_tr):.2f}GB  seizures={y_tr.sum()}")
    print(f"  X_val:   {X_val.shape}  {gb(X_val):.2f}GB  seizures={y_val.sum()}")
    print(f"  X_test:  {X_te.shape}  {gb(X_te):.2f}GB  seizures={y_te.sum()}")
    print(f"  Peak 3ch output: {gb(X_te)*3*3:.2f}GB (largest split × 3 channels)")

    # ── Add frequency channels (split by split, free between) ─────────────────
    print(f"\nAdding frequency channels to train split ({len(X_tr)} windows)...")
    X_tr_3ch  = add_frequency_channels(X_tr,  label='train')
    del X_tr

    print(f"\nAdding frequency channels to val split ({len(X_val)} windows)...")
    X_val_3ch = add_frequency_channels(X_val, label='val')
    del X_val

    print(f"\nAdding frequency channels to test split ({len(X_te)} windows)...")
    X_te_3ch  = add_frequency_channels(X_te,  label='test')
    del X_te

    # ── Scale: fit on train (ch0 already [0,255]; ch1/ch2 need scaling) ───────
    # ch0 (raw EEG) is already [0,255] from dataset_ann.npz, so its scaler
    # will map it to [0,255] with effectively identity (min≈0, max≈255).
    # ch1/ch2 (envelopes) are in physical units — scaler normalises them.
    print(f"\nFitting per-channel scaler on train split...")
    scaler = fit_scaler(X_tr_3ch)
    ch_names = {0: 'raw EEG (already [0,255])', 1: 'delta env', 2: 'beta env'}
    for ch in range(3):
        print(f"  ch{ch} {ch_names[ch]}: "
              f"[{scaler[f'ch{ch}_min']:.3f}, {scaler[f'ch{ch}_max']:.3f}] → [0,255]")

    print("Scaling all splits to [0, 255]...")
    X_tr_sc  = apply_scaler(X_tr_3ch,  scaler); del X_tr_3ch
    X_val_sc = apply_scaler(X_val_3ch, scaler); del X_val_3ch
    X_te_sc  = apply_scaler(X_te_3ch,  scaler); del X_te_3ch

    assert X_tr_sc.min() >= 0.0 and X_tr_sc.max() <= 255.01, "Scale error on train"
    print(f"  Scaled range check: [{X_tr_sc.min():.1f}, {X_tr_sc.max():.1f}]  ✓")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path    = os.path.join(data_dir, f'{patient}_dataset_stft.npz')
    scaler_path = os.path.join(data_dir, f'{patient}_scaler_stft.json')

    print(f"\nSaving {out_path} ...")
    np.savez_compressed(out_path,
                        X_train=X_tr_sc,  y_train=y_tr,
                        X_val=X_val_sc,   y_val=y_val,
                        X_test=X_te_sc,   y_test=y_te)
    with open(scaler_path, 'w') as f:
        json.dump(scaler, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE — {patient}")
    print(f"  {out_path}")
    print(f"  X_train: {X_tr_sc.shape}   [{X_tr_sc.min():.0f},{X_tr_sc.max():.0f}]")
    print(f"  X_val:   {X_val_sc.shape}  [{X_val_sc.min():.0f},{X_val_sc.max():.0f}]")
    print(f"  X_test:  {X_te_sc.shape}  [{X_te_sc.min():.0f},{X_te_sc.max():.0f}]")
    print(f"  Test seizures: {y_te.sum()}")
    print(f"  Shape check:   {X_tr_sc.shape[1:]}  (expect (18, 512, 3))")
    print(f"{'='*60}")


# ── Multi-patient pipeline ─────────────────────────────────────────────────────

def build_multi_patient(patients: list = MULTI_PATIENTS,
                        data_dir: str = 'data/processed/') -> None:
    """
    Pool train+val splits from all patients.
    Test split not included — use per-patient _dataset_stft.npz for evaluation.
    Processes one patient at a time; frees memory between patients.
    """
    print(f"\n{'='*60}")
    print(f"Building multi-patient STFT dataset: {patients}")
    print(f"{'='*60}")

    all_train_X, all_train_y = [], []
    all_val_X,   all_val_y   = [], []
    per_patient_scalers = {}

    for patient in patients:
        ann_path = os.path.join(data_dir, f'{patient}_dataset_ann.npz')
        if not os.path.exists(ann_path):
            print(f"ERROR: {ann_path} not found.")
            sys.exit(1)

        print(f"\n{'─'*40}")
        print(f"Processing {patient} ...")
        data  = np.load(ann_path)
        X_tr  = data['X_train'].astype(np.float32)
        y_tr  = data['y_train']
        X_val = data['X_val'].astype(np.float32)
        y_val = data['y_val']

        print(f"  train: {X_tr.shape}  val: {X_val.shape}")

        X_tr_3ch  = add_frequency_channels(X_tr,  label=f'{patient}/tr');  del X_tr
        X_val_3ch = add_frequency_channels(X_val, label=f'{patient}/val'); del X_val

        scaler = fit_scaler(X_tr_3ch)
        per_patient_scalers[patient] = scaler

        X_tr_sc  = apply_scaler(X_tr_3ch,  scaler); del X_tr_3ch
        X_val_sc = apply_scaler(X_val_3ch, scaler); del X_val_3ch

        all_train_X.append(X_tr_sc)
        all_train_y.append(y_tr)
        all_val_X.append(X_val_sc)
        all_val_y.append(y_val)
        print(f"  {patient} done  train={X_tr_sc.shape}  val={X_val_sc.shape}")

    print(f"\nPooling {len(patients)} patients...")
    X_train_pool = np.concatenate(all_train_X, axis=0); del all_train_X
    y_train_pool = np.concatenate(all_train_y, axis=0)
    X_val_pool   = np.concatenate(all_val_X,   axis=0); del all_val_X
    y_val_pool   = np.concatenate(all_val_y,   axis=0)

    print(f"  Pooled train: {X_train_pool.shape}  seizures={y_train_pool.sum()}")
    print(f"  Pooled val:   {X_val_pool.shape}    seizures={y_val_pool.sum()}")

    out_path    = os.path.join(data_dir, 'multi_dataset_stft.npz')
    scaler_path = os.path.join(data_dir, 'multi_scaler_stft.json')

    print(f"\nSaving {out_path} ...")
    np.savez_compressed(out_path,
                        X_train=X_train_pool, y_train=y_train_pool,
                        X_val=X_val_pool,     y_val=y_val_pool)
    with open(scaler_path, 'w') as f:
        json.dump(per_patient_scalers, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE — multi-patient STFT dataset")
    print(f"  {out_path}")
    print(f"  X_train: {X_train_pool.shape}")
    print(f"  X_val:   {X_val_pool.shape}")
    print(f"  Use per-patient stft npz for test evaluation")
    print(f"{'='*60}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Build 3-channel STFT dataset.')
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--patient',       type=str)
    group.add_argument('--multi-patient', action='store_true')
    parser.add_argument('--data-dir',     type=str, default='data/processed/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    try:
        import scipy; print(f"scipy {scipy.__version__} ✓")
    except ImportError:
        print("ERROR: pip install scipy --break-system-packages"); sys.exit(1)

    if args.multi_patient:
        build_multi_patient(data_dir=args.data_dir)
    else:
        build_single_patient(patient=args.patient, data_dir=args.data_dir)
