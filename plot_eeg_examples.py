"""
plot_eeg_examples.py
========================
Plots stacked multi-channel EEG traces (standard clinical montage style)
to illustrate the physiological-heterogeneity argument visually: baseline
vs. seizure for one patient, then a seizure from a contrasting patient,
so the morphological difference is directly visible rather than argued
from numbers alone.

Designed around this project's actual data shape: (18 channels, 512
samples), 512 samples / 2.0s window = 256 Hz (CHB-MIT's native rate).
Reads the RAW (unscaled) per-patient windows -- data/processed/
<patient>_X.npy / _y.npy -- same files build_dataset_lopo_fold.py and
build_lopo_eval_set.py already read, so if those exist (they do, for all
15 roster patients, confirmed this session) this needs no new
preprocessing.

Suggested pairing for the dissertation: chb10 (clean LOPO fold, collapse
PASS, sensible seizure morphology this project's model reads well) vs.
chb13 (documented hard patient across DANN/CORAL/SSL/G AND LOPO --
consistently the hardest patient in this project by a different failure
mode each time). A reader seeing chb10's and chb13's seizure traces side
by side, both correctly labelled "seizure" by the dataset but visually
quite different in morphology, is a strong intuitive argument for why a
single shared representation struggles -- exactly the physiological
heterogeneity point in the discussion chapter.

Usage:
    python3 plot_eeg_examples.py --patient-a chb10 --patient-b chb13
    python3 plot_eeg_examples.py --patient-a chb10 --patient-b chb13 --n-channels 8

Output (in figures/):
    eeg_<patient>_baseline.png       — one representative non-seizure window
    eeg_<patient>_seizure.png        — one representative seizure window
    eeg_seizure_comparison_<a>_vs_<b>.png  — both patients' seizures side by side
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--patient-a', default='chb10', help='"Clean" comparison patient')
parser.add_argument('--patient-b', default='chb13', help='"Hard" comparison patient')
parser.add_argument('--n-channels', type=int, default=8,
                    help="How many of the 18 channels to plot (all 18 is "
                         "legible in a wide figure; 6-8 is more readable "
                         "for a dissertation page). Default 8.")
parser.add_argument('--fs', type=float, default=256.0,
                    help="Sampling rate in Hz (512 samples / 2.0s window "
                         "= 256 Hz, CHB-MIT's native rate -- default matches "
                         "this project's window_s=2.0 default).")
parser.add_argument('--out-dir', default='figures')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


def load_patient(patient):
    X_path = f'data/processed/{patient}_X.npy'
    y_path = f'data/processed/{patient}_y.npy'
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"{X_path} / {y_path} not found. These are the same raw "
            f"windows build_dataset_lopo_fold.py reads -- if the LOPO "
            f"sweep ran successfully for this patient's roster position, "
            f"they should already be on disk. If not: "
            f"python3 src/preprocessing/preprocess.py --patient {patient}")
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path)
    return X, y


def pick_representative_window(X, y, want_seizure):
    """
    Picks the middle-most window of the requested class, rather than the
    first match -- the first seizure window is often right at onset
    (partially baseline), the middle of a seizure run is more visually
    representative of established ictal morphology. Same idea for
    baseline: avoids edge-of-recording artifacts.
    """
    idx = np.where(y == (1 if want_seizure else 0))[0]
    if len(idx) == 0:
        raise ValueError(f"No {'seizure' if want_seizure else 'baseline'} "
                          f"windows found in this patient's data.")
    return int(idx[len(idx) // 2])


def plot_stacked(X_window, fs, n_channels, title, out_path, seizure_shading=False):
    """
    Standard clinical-style stacked montage: each channel vertically
    offset, shared time axis. X_window shape: (18, 512) or (n_ch, 512).
    """
    n_ch_total = X_window.shape[0]
    n_ch = min(n_channels, n_ch_total)
    n_samples = X_window.shape[1]
    t = np.arange(n_samples) / fs

    fig, ax = plt.subplots(figsize=(9, 0.55 * n_ch + 1.2))
    offset_step = 4.0
    for ch in range(n_ch):
        trace = X_window[ch]
        trace_norm = (trace - trace.mean())
        std = trace_norm.std()
        if std > 1e-8:
            trace_norm = trace_norm / std
        y_offset = (n_ch - ch) * offset_step
        color = '#C62828' if seizure_shading else '#1565C0'
        ax.plot(t, trace_norm + y_offset, color=color, linewidth=0.6)

    ax.set_yticks([(n_ch - ch) * offset_step for ch in range(n_ch)])
    ax.set_yticklabels([f'Ch{ch+1}' for ch in range(n_ch)], fontsize=8)
    ax.set_xlabel('Time (s)')
    ax.set_title(title, fontsize=10)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


for patient in (args.patient_a, args.patient_b):
    X, y = load_patient(patient)

    baseline_idx = pick_representative_window(X, y, want_seizure=False)
    seizure_idx = pick_representative_window(X, y, want_seizure=True)

    plot_stacked(
        np.array(X[baseline_idx]), args.fs, args.n_channels,
        f'{patient} — representative baseline (interictal) window',
        os.path.join(args.out_dir, f'eeg_{patient}_baseline.png'),
        seizure_shading=False)

    plot_stacked(
        np.array(X[seizure_idx]), args.fs, args.n_channels,
        f'{patient} — representative seizure (ictal) window',
        os.path.join(args.out_dir, f'eeg_{patient}_seizure.png'),
        seizure_shading=True)

# ── Side-by-side seizure comparison (the key illustrative figure) ──────────
X_a, y_a = load_patient(args.patient_a)
X_b, y_b = load_patient(args.patient_b)
idx_a = pick_representative_window(X_a, y_a, want_seizure=True)
idx_b = pick_representative_window(X_b, y_b, want_seizure=True)

n_ch = min(args.n_channels, X_a.shape[1] if X_a.ndim > 1 else args.n_channels)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 0.55 * n_ch + 1.5), sharey=True)
offset_step = 4.0
t = np.arange(X_a.shape[-1]) / args.fs

for ax, X, idx, patient in [(ax1, X_a, idx_a, args.patient_a), (ax2, X_b, idx_b, args.patient_b)]:
    window = np.array(X[idx])
    for ch in range(n_ch):
        trace = window[ch]
        trace_norm = trace - trace.mean()
        std = trace_norm.std()
        if std > 1e-8:
            trace_norm = trace_norm / std
        y_offset = (n_ch - ch) * offset_step
        ax.plot(t, trace_norm + y_offset, color='#C62828', linewidth=0.6)
    ax.set_title(f'{patient} — seizure', fontsize=10)
    ax.set_xlabel('Time (s)')
    ax.spines[['top', 'right', 'left']].set_visible(False)

ax1.set_yticks([(n_ch - ch) * offset_step for ch in range(n_ch)])
ax1.set_yticklabels([f'Ch{ch+1}' for ch in range(n_ch)], fontsize=8)
fig.suptitle(f'Seizure morphology: {args.patient_a} vs. {args.patient_b} — '
             'same label, visually distinct ictal patterns', fontsize=12)
fig.tight_layout()
out_path = os.path.join(args.out_dir, f'eeg_seizure_comparison_{args.patient_a}_vs_{args.patient_b}.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")

print()
print("Suggested caption/discussion text (adapt after actually looking at "
      "the output — don't describe features that aren't visibly there):")
print(f"  \"Figure X compares representative seizure-window EEG traces from "
      f"{args.patient_a} and {args.patient_b}. Despite sharing the same "
      f"binary label under this project's window-level annotation scheme, "
      f"the two traces show visually distinct morphology [describe: "
      f"amplitude, rhythmicity, which channels are most active, onset "
      f"sharpness -- fill in from what's actually visible]. This is "
      f"consistent with the LOPO finding that {args.patient_b} fails "
      f"under cross-patient generalisation regardless of which technique "
      f"is used to try to fix it (Section X), suggesting the limitation "
      f"is physiological rather than purely a modelling or data-volume "
      f"problem.\"")
