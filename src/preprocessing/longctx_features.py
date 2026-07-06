"""
src/preprocessing/longctx_features.py
======================================
Causal, long-lookback rolling features for the Gate 1 long-context channels
(Handoff_architecture_scoping_to_implementation.md §4).

Both functions take the CONTINUOUS per-file signal (post bandpass/notch/CAR,
pre-windowing) and return an array of the same shape, so the caller can window
them with the identical start/step loop used for the raw EEG channel —
guaranteeing identical window boundaries across all 3 channels.

Causality / file-boundary rule (shared by both functions):
  The first `lookback_samples - 1` positions of each file don't have a full
  trailing window available. Rather than compute a growing/partial window
  for them (which would silently leak a different, shorter-lookback feature
  at the start of every file) or look backward across the file boundary,
  they are padded with the FIRST fully-valid value. This is a flat constant
  region of length `lookback_samples - 1` (~12s) at the start of every file's
  feature stream — small relative to a ~60min recording, and never crosses
  a file boundary.
"""
import numpy as np
from scipy.signal import sosfilt

from src.preprocessing.build_dataset_stft import butter_bandpass_sos


def _causal_trailing_sum(arr: np.ndarray, lookback_samples: int) -> np.ndarray:
    """
    Vectorised (cumsum-based) trailing sum over `lookback_samples`, per row.

    arr: (n_channels, n_timepoints)
    Returns: (n_channels, n_timepoints) — out[:, t] = sum(arr[:, t-L+1 : t+1])
             for t >= L-1; padded with out[:, L-1] for t < L-1.
    """
    n_ch, n_t = arr.shape
    L = int(lookback_samples)
    assert 1 <= L < n_t, f"lookback_samples={L} invalid for signal length {n_t}"

    cumsum = np.cumsum(arr, axis=1, dtype=np.float64)
    shifted = np.zeros_like(cumsum)
    shifted[:, L:] = cumsum[:, :n_t - L]
    trailing = cumsum - shifted  # full L-window from t=L-1 onward

    trailing[:, :L - 1] = trailing[:, L - 1:L]  # pad with first valid value
    return trailing


def rolling_line_length(signal: np.ndarray, lookback_samples: int) -> np.ndarray:
    """
    Causal rolling line-length: sum(|x[t]-x[t-1]|) over a trailing window of
    `lookback_samples`, per channel, same length as the input signal.

    signal: (n_channels, n_timepoints) float32/float64 — continuous per-file
    Returns: (n_channels, n_timepoints) float32
    """
    diffs = np.abs(np.diff(signal, axis=1, prepend=signal[:, :1]))  # diffs[:,0]=0
    trailing = _causal_trailing_sum(diffs, lookback_samples)
    return trailing.astype(np.float32)


def rolling_band_ratio(signal: np.ndarray, lookback_samples: int, fs: float = 256,
                        low_band: tuple = (0.5, 4.0),
                        high_band: tuple = (13.0, 30.0)) -> np.ndarray:
    """
    Causal rolling delta/beta RMS power ratio over a trailing window of
    `lookback_samples`. Reuses the same band-pass machinery already in
    build_dataset_stft.py (0.5-4Hz / 13-30Hz) — same filters, longer RMS
    window than the existing per-window envelope.

    signal: (n_channels, n_timepoints) float32/float64 — continuous per-file
    Returns: (n_channels, n_timepoints) float32 — delta_rms / (beta_rms + eps)
    """
    sos_low = butter_bandpass_sos(*low_band, fs=fs)
    sos_high = butter_bandpass_sos(*high_band, fs=fs)

    # Causal filtering (sosfilt, not sosfiltfilt) — must not use future
    # samples, matching the "causal" framing of the lookback itself.
    delta = sosfilt(sos_low, signal, axis=1)
    beta = sosfilt(sos_high, signal, axis=1)

    L = int(lookback_samples)
    delta_power = _causal_trailing_sum(delta ** 2, L) / L
    beta_power = _causal_trailing_sum(beta ** 2, L) / L

    eps = 1e-8
    ratio = np.sqrt(delta_power) / (np.sqrt(beta_power) + eps)
    return ratio.astype(np.float32)
