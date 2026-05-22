"""
CHB-MIT Dataset EDA
Inspects signal quality, frequency content, and class balance.

Run from: ~/dissertation/
    python3 code/preprocessing/eda.py

Outputs saved to: results/
"""
import os
import re
import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
# Adjust this path to wherever wget saved the files
DATA_DIR    = 'data/raw/chbmit/physionet.org/files/chbmit/1.0.0/chb01/'
EDF_FILE    = os.path.join(DATA_DIR, 'chb01_01.edf')   # no seizures — clean look
SUMMARY     = os.path.join(DATA_DIR, 'chb01-summary.txt')
RESULTS_DIR = 'results/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Basic file properties ──────────────────────────────────────────────────
print("=" * 60)
print("1. FILE PROPERTIES")
print("=" * 60)

raw = mne.io.read_raw_edf(EDF_FILE, preload=True, verbose=False)
sfreq      = raw.info['sfreq']
n_channels = len(raw.ch_names)
duration_s = raw.times[-1]

print(f"Sampling rate : {sfreq} Hz")
print(f"Channels      : {n_channels}")
print(f"Duration      : {duration_s:.1f} s  ({duration_s/60:.1f} min)")
print(f"Channel names : {raw.ch_names}")
print(f"Expected shape: ({n_channels}, {int(duration_s * sfreq)}) samples")

# ── 2. Check for the 18 common channels ──────────────────────────────────────
# These 18 channels are present in all CHB-MIT patients.
# Using only these ensures consistent model input shape across patients.
# Source: Shoeb (2009) PhD thesis, Appendix A
COMMON_CHANNELS = [
    'FP1-F7', 'F7-T7',  'T7-P7',  'P7-O1',
    'FP1-F3', 'F3-C3',  'C3-P3',  'P3-O1',
    'FP2-F4', 'F4-C4',  'C4-P4',  'P4-O2',
    'FP2-F8', 'F8-T8',  'T8-P8',  'P8-O2',
    'FZ-CZ',  'CZ-PZ'
]

print("\n" + "=" * 60)
print("2. CHANNEL AVAILABILITY CHECK")
print("=" * 60)
present = [ch for ch in COMMON_CHANNELS if ch in raw.ch_names]
missing = [ch for ch in COMMON_CHANNELS if ch not in raw.ch_names]
print(f"Common channels present : {len(present)}/18")
if missing:
    print(f"Missing                 : {missing}")
    print("NOTE: record this in your dissertation methodology section")
else:
    print("All 18 common channels present. ✓")

# ── 3. Raw signal plot (4 channels, 10 s window) ─────────────────────────────
print("\n" + "=" * 60)
print("3. RAW SIGNAL PLOT")
print("=" * 60)

raw_common = raw.copy().pick_channels(present)
data, times = raw_common[:4, int(sfreq*60):int(sfreq*70)]  # minute 1–1:10

fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(times, data[i] * 1e6, linewidth=0.5, color='steelblue')
    ax.set_ylabel(present[i], fontsize=8, rotation=0, ha='right', va='center')
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[-1].set_xlabel('Time (s)')
plt.suptitle(f'CHB-MIT chb01_01.edf — Raw EEG (4 channels, 10 s)', y=1.01)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'eda_raw_signal.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()

# Copy to Windows desktop for viewing
win_user = os.environ.get('WIN_USER', 'YourWindowsUsername')
win_path = f'/mnt/c/Users/{win_user}/Desktop/eda_raw_signal.png'
os.system(f'cp {out} "{win_path}" 2>/dev/null')

# ── 4. Power spectral density ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. POWER SPECTRAL DENSITY")
print("=" * 60)
fig = raw_common.plot_psd(fmax=80, show=False, verbose=False)
out = os.path.join(RESULTS_DIR, 'eda_psd.png')
fig.savefig(out, dpi=120)
plt.close('all')
print(f"Saved: {out}")
print("What to look for:")
print("  - Peak around 10 Hz (alpha) = normal resting EEG")
print("  - Spike at 60 Hz = mains interference (US recording) — notch filter needed")
print("  - Flat/noisy spectrum = artefact or bad electrode")

# ── 5. Class balance across ALL chb01 files ───────────────────────────────────
print("\n" + "=" * 60)
print("5. CLASS BALANCE (all chb01 files)")
print("=" * 60)

def parse_chbmit_summary(summary_path):
    """
    Parse CHB-MIT summary file.

    Returns dict: { 'chb01_03.edf': [(start_s, end_s), ...], ... }

    IMPORTANT: Times are seconds from the START of each EDF file,
    not from the recording session start. This is critical for
    correct window labelling.

    Source: CHB-MIT database documentation, PhysioNet.
    https://physionet.org/content/chbmit/1.0.0/
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
                # Handle both "Seizure Start Time:" and "Seizure N Start Time:"
                start_s = int(re.search(r'(\d+)\s+second', start_line).group(1))
                end_s   = int(re.search(r'(\d+)\s+second', end_line).group(1))
                seizures[current_file].append((start_s, end_s))
        i += 1

    return seizures

seizure_map = parse_chbmit_summary(SUMMARY)

total_s, seizure_s = 0, 0
for fname, intervals in seizure_map.items():
    edf_path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(edf_path):
        continue
    r = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    dur = r.times[-1]
    total_s += dur
    for (s, e) in intervals:
        seizure_s += (e - s)

non_seizure_s = total_s - seizure_s
ratio = int(non_seizure_s / max(seizure_s, 1))

print(f"Total recording  : {total_s/3600:.2f} hours")
print(f"Seizure time     : {seizure_s:.1f} s  ({100*seizure_s/total_s:.2f}%)")
print(f"Non-seizure time : {non_seizure_s:.1f} s  ({100*non_seizure_s/total_s:.2f}%)")
print(f"Imbalance ratio  : 1 seizure window : ~{ratio} non-seizure windows")
print()
print(">> SMOTE oversampling OR class-weighted loss is mandatory.")
print(">> A model predicting all non-seizure would score ~98% accuracy.")
print(">> Use SENSITIVITY as your primary metric, not accuracy.")

# ── 6. Seizure event inventory ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. SEIZURE EVENT INVENTORY")
print("=" * 60)
for fname, intervals in sorted(seizure_map.items()):
    if intervals:
        for (s, e) in intervals:
            print(f"  {fname}  →  {s}s – {e}s  (duration: {e-s}s)")

n_total = sum(len(v) for v in seizure_map.values())
print(f"\nTotal seizures in chb01: {n_total}")
print("Note: chb01 is paediatric female, age ~11.")
print("Seizure morphology varies per patient — train/test on same patient first.")

print("\n=== EDA complete. Review plots before running preprocessing. ===")
