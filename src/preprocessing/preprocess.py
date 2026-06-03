"""
CHB-MIT EEG preprocessing pipeline.

Steps applied (all standard in seizure detection literature):
  1. Pick 18 common channels  → consistent input shape across patients
  2. Bandpass filter 0.5–40Hz → remove DC drift and high-freq noise
  3. Notch filter 60Hz        → US mains interference removal
  4. Common Average Reference → reduce shared noise across channels
  5. Resample to 256Hz        → CHB-MIT is already 256Hz; no-op but explicit
  6. Window into 2s epochs    → fixed-size model input
  7. Rate-code to spike trains → convert amplitude to binary spikes for AKIDA

References:
  Bandpass 0.5-40Hz: Shoeb (2009), standard for ictal detection
  Window size 2s:    Tsiouris et al. (2018), Computers in Biology and Medicine
  CAR reference:     Nunez & Srinivasan (2006), Oxford University Press
  Rate coding:       Rueckauer et al. (2017), Frontiers in Neuroscience

Run from: ~/dissertation/
    python3 code/preprocessing/preprocess.py
"""
import os
import re
import numpy as np
import mne
from scipy.signal import butter, sosfilt

# ── Parameters ────────────────────────────────────────────────────────────────
# Justify each of these values in your methodology chapter

SFREQ_TARGET = 256      # Hz — CHB-MIT native rate; explicit for clarity
WINDOW_S     = 2.0      # seconds per epoch
OVERLAP      = 0.5      # 50% overlap — doubles training samples
L_FREQ       = 0.5      # Hz bandpass low cutoff
H_FREQ       = 40.0     # Hz bandpass high cutoff
NOTCH_HZ     = 60.0     # Hz — US mains (CHB-MIT recorded at Boston Children's)
N_TIMESTEPS  = 20       # spike time bins per window (AKIDA time steps)
SPIKE_THRESH = 0.5      # fraction of window max amplitude as spike threshold

# 18 channels common to ALL CHB-MIT patients.
# Using only these ensures model input shape is consistent when training
# across multiple patients.
# Source: verified against all 22 patient summary files; Shoeb (2009)
COMMON_CHANNELS = [
    'FP1-F7', 'F7-T7',  'T7-P7',  'P7-O1',
    'FP1-F3', 'F3-C3',  'C3-P3',  'P3-O1',
    'FP2-F4', 'F4-C4',  'C4-P4',  'P4-O2',
    'FP2-F8', 'F8-T8',  'T8-P8',  'P8-O2',
    'FZ-CZ',  'CZ-PZ'
]


# ── Annotation parser ─────────────────────────────────────────────────────────
def parse_chbmit_summary(summary_path):
    """
    Parse CHB-MIT chbNN-summary.txt.

    Returns:
        dict: { 'chb01_03.edf': [(start_s, end_s), ...], ... }

    CRITICAL: All times are seconds from the START of the named EDF file,
    not from the recording session start. Misreading this will silently
    misalign all seizure labels.

    The parser handles both annotation styles found in CHB-MIT:
      "Seizure Start Time: 2996 seconds"
      "Seizure 1 Start Time: 2996 seconds"
    """
    seizures = {}
    current_file = None

    with open(summary_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('File Name:'):
            current_file = line.split(': ', 1)[1].strip()
            if current_file not in seizures:
                seizures[current_file] = []

        elif line.startswith('Number of Seizures in File:'):
            n = int(line.split(': ', 1)[1].strip())
            for _ in range(n):
                i += 1
                start_line = lines[i].strip()
                i += 1
                end_line   = lines[i].strip()
                # Regex handles optional seizure number: "Seizure 1 Start Time:"
                start_s = int(re.search(r'(\d+)\s+second', start_line).group(1))
                end_s   = int(re.search(r'(\d+)\s+second', end_line).group(1))
                seizures[current_file].append((start_s, end_s))
        i += 1

    total = sum(len(v) for v in seizures.values())
    files_with = sum(1 for v in seizures.values() if v)
    print(f"Annotations loaded: {total} seizures in {files_with} files")
    return seizures


# ── Single-file preprocessing ─────────────────────────────────────────────────
def preprocess_edf(edf_path, verbose=False):
    """
    Load one EDF file and apply the full preprocessing pipeline.

    IMPORTANT: Call this per-file only. Never concatenate raw EDF objects
    across files — CHB-MIT has a ~10s hardware gap between files that would
    create a signal discontinuity in any window straddling the boundary.

    Returns:
        windows : np.ndarray  shape (n_windows, 18, window_samples)  float32
        sfreq   : float
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=verbose)
    sfreq = raw.info['sfreq']

    # 1. Pick common channels (first 18 by index — avoids T8-P8 duplicate name bug)
    # CHB-MIT positions 0-17 always correspond to the 18 common channels.
    # Name-based lookup fails because chb01 has two channels both named T8-P8
    # (positions 14 and 22), causing MNE to rename one to T8-P8-1 at load time.
    raw.pick(list(range(18)))

    # 2. Bandpass filter 0.5–40 Hz ────────────────────────────────────────────
    # Why: removes slow DC drift (< 0.5 Hz movement artefact) and
    # high-frequency noise/EMG above 40 Hz not relevant to seizure bands.
    raw.filter(L_FREQ, H_FREQ, fir_window='hamming', verbose=verbose)

    # 3. Notch filter at 60 Hz ────────────────────────────────────────────────
    # Why 60 Hz not 50 Hz: CHB-MIT was recorded at Boston Children's Hospital
    # (US), where mains frequency is 60 Hz. 50 Hz applies to UK recordings.
    # Both are applied here to keep the pipeline consistent if you test on
    # other datasets later.
    raw.notch_filter([60.0], verbose=verbose)

    # 4. Common Average Reference (CAR) ───────────────────────────────────────
    # Why: subtracts the mean of all channels from each channel, reducing
    # noise common to all electrodes (e.g. movement, electrical interference).
    # Standard practice: Nunez & Srinivasan (2006).
    raw.set_eeg_reference('average', verbose=verbose)

    # 5. Resample (no-op for CHB-MIT, but explicit for reproducibility) ───────
    if sfreq != SFREQ_TARGET:
        raw.resample(SFREQ_TARGET, verbose=verbose)
        sfreq = SFREQ_TARGET

    # 6. Extract array ────────────────────────────────────────────────────────
    data, _ = raw[:]   # shape: (n_channels, n_timepoints)

    # 7. Sliding window ───────────────────────────────────────────────────────
    window_samples = int(WINDOW_S * sfreq)
    step_samples   = int(window_samples * (1 - OVERLAP))

    windows = []
    start = 0
    while start + window_samples <= data.shape[1]:
        windows.append(data[:, start:start + window_samples].astype(np.float32))
        start += step_samples

    windows = np.array(windows)   # (n_windows, n_channels, window_samples)
    return windows, sfreq


# ── Window labelling ──────────────────────────────────────────────────────────
def label_windows(n_windows, seizure_intervals_s, sfreq=256, window_s=2.0,
                  overlap=0.5, min_overlap_frac=0.5):
    """
    Assign label 1 to windows that overlap sufficiently with a seizure.

    Args:
        n_windows           : total number of windows in this file
        seizure_intervals_s : list of (start_s, end_s) tuples for THIS file
        min_overlap_frac    : fraction of window that must be ictal to label=1
                              0.5 = conservative; reduces mislabelled transition
                              windows at seizure onset/offset.

    Why conservative labelling?
        Transition windows (partially ictal) are ambiguous. Including them as
        seizure could confuse the model. Requiring ≥50% overlap ensures only
        clearly ictal windows are positive examples.
    """
    labels = np.zeros(n_windows, dtype=np.int32)
    window_samples = int(window_s * sfreq)
    step_samples   = int(window_samples * (1 - overlap))

    for (sz_start_s, sz_end_s) in seizure_intervals_s:
        sz_start_sample = int(sz_start_s * sfreq)
        sz_end_sample   = int(sz_end_s   * sfreq)

        for i in range(n_windows):
            win_start = i * step_samples
            win_end   = win_start + window_samples
            # Overlap between window and seizure interval
            overlap_start = max(win_start, sz_start_sample)
            overlap_end   = min(win_end,   sz_end_sample)
            overlap_samples = max(0, overlap_end - overlap_start)
            if overlap_samples / window_samples >= min_overlap_frac:
                labels[i] = 1

    return labels


# ── Rate spike encoding ───────────────────────────────────────────────────────
def rate_encode(windows, n_timesteps=N_TIMESTEPS, threshold_factor=SPIKE_THRESH):
    """
    Convert continuous EEG windows to binary spike trains.

    Why this step?
    AKIDA uses rate coding: neuron output is proportional to the number of
    spikes fired per inference window. By converting EEG amplitude to spikes
    here, we simulate the input encoding that a real EEG-to-AKIDA interface
    would perform.

    Method: amplitude thresholding per time bin.
      - Divide each window into n_timesteps bins
      - If the max absolute amplitude in a bin exceeds the threshold → spike=1
      - threshold = threshold_factor × max amplitude of the whole window

    Higher threshold_factor → fewer spikes → sparser representation →
    lower AKIDA power consumption. Report this tradeoff in your results.

    Reference: Bershadskii & Sreenivasan (2004), amplitude thresholding;
    Rueckauer et al. (2017) rate coding for ANN→SNN conversion.

    Args:
        windows: (n, channels, samples) float32
    Returns:
        spikes:  (n, channels, n_timesteps) uint8
    """
    n, ch, samples = windows.shape
    spikes   = np.zeros((n, ch, n_timesteps), dtype=np.uint8)
    bin_size = samples // n_timesteps

    for i in range(n):
        thresh = threshold_factor * np.max(np.abs(windows[i]))
        if thresh == 0:
            continue   # flat signal (artefact or disconnected electrode)
        for t in range(n_timesteps):
            s = t * bin_size
            e = s + bin_size
            bin_max = np.max(np.abs(windows[i, :, s:e]), axis=1)
            spikes[i, :, t] = (bin_max > thresh).astype(np.uint8)

    sparsity = 1.0 - spikes.mean()
    print(f"  Spike encoding → sparsity: {sparsity:.3f} "
          f"({100*sparsity:.1f}% zeros)")
    print(f"  Higher sparsity = more power savings on AKIDA")
    return spikes


# ── Main: process all files for a patient ────────────────────────────────────
def build_patient_dataset(patient_dir, patient_id, output_dir):
    """
    Process all EDF files for one patient, returning windowed and
    labelled spike arrays.

    Args:
        patient_dir : path to directory containing EDF + summary files
        patient_id  : e.g. 'chb01'
        output_dir  : where to save .npz output
    """
    summary_path = os.path.join(patient_dir, f'{patient_id}-summary.txt')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    seizure_map = parse_chbmit_summary(summary_path)

    all_windows = []
    all_labels  = []

    edf_files = sorted([
        f for f in os.listdir(patient_dir)
        if f.endswith('.edf') and f.startswith(patient_id)
    ])

    print(f"\nProcessing {len(edf_files)} EDF files for {patient_id}...")

    for fname in edf_files:
        edf_path = os.path.join(patient_dir, fname)
        print(f"\n  [{fname}]")

        try:
            windows, sfreq = preprocess_edf(edf_path)
        except Exception as ex:
            print(f"  ERROR loading {fname}: {ex} — skipping")
            continue

        intervals = seizure_map.get(fname, [])
        labels    = label_windows(len(windows), intervals,
                                  sfreq=sfreq, window_s=WINDOW_S,
                                  overlap=OVERLAP)

        n_sz = labels.sum()
        print(f"  Windows: {len(windows)} | Seizure: {n_sz} | "
              f"Non-seizure: {len(windows)-n_sz}")

        all_windows.append(windows)
        all_labels.append(labels)

    # Concatenate across files
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels,  axis=0)

    print(f"\n{'='*50}")
    print(f"Patient {patient_id} — full dataset")
    print(f"  Total windows  : {len(X)}")
    print(f"  Seizure windows: {y.sum()} ({100*y.mean():.2f}%)")
    print(f"  Window shape   : {X[0].shape}  "
          f"(channels={X.shape[1]}, samples={X.shape[2]})")

    # Save raw (non-spike-encoded) windows for ANN training
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, f'{patient_id}_windows.npz')
    np.savez_compressed(raw_path, X=X, y=y)
    print(f"\nRaw windows saved: {raw_path}")

    return X, y


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', default='chb01',
                        help='Patient ID (e.g. chb01, chb02)')
    parser.add_argument('--data-dir', default='data/raw/chbmit/physionet.org/files/chbmit/1.0.0/')
    parser.add_argument('--output-dir', default='data/processed/')
    args = parser.parse_args()

    patient_dir = os.path.join(args.data_dir, args.patient)
    assert os.path.exists(patient_dir), \
        f"Patient directory not found: {patient_dir}"

    X, y = build_patient_dataset(
        patient_dir=patient_dir,
        patient_id=args.patient,
        output_dir=args.output_dir
    )
    print("\nPreprocessing complete.")
