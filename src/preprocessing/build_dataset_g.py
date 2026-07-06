"""
src/preprocessing/build_dataset_g.py
=====================================
Candidate G — Phase 1 (Handoff_a_e_c_closed_to_next_steps.md sec8):
physiologically-normalised classical features (relative band power +
delta/beta ratio), REPLACING raw EEG entirely -- not augmenting it.

WHY REPLACE, NOT AUGMENT (design decision, flagged per project norm of
disclosing forks before building rather than after)
--------------------------------------------------------------------------
Per sec8's own framing, G is interesting specifically because hand-
engineered features "bypass the CNN's learned representation entirely."
Keeping ch0 = raw EEG (as build_dataset_stft.py's 3ch design did for
Candidate STFT/Gate 2) would let the CNN fall back on learning from raw
amplitude whenever that's easier for the loss -- weakening the very test
this candidate exists to run. All three channels here are ratios /
normalised quantities; none carries raw amplitude information. If G
still fails on chb13 the same way DANN/CORAL/SSL did, that's a genuinely
stronger negative result than if raw EEG were still in the mix and could
be quietly doing the work.

THREE CHANNELS (all classical, all scale-free)
------------------------------------------------
  ch0: relative delta power = delta_env / (total_env + EPS)
  ch1: relative beta power  = beta_env  / (total_env + EPS)
  ch2: delta/beta ratio     = clip(delta_env / (beta_env + EPS), 0, ratio_cap)

  delta_env, beta_env: reused UNCHANGED from build_dataset_stft.py's
  compute_band_envelope() -- same 0.5-4Hz / 13-30Hz filters, same 64-sample
  RMS window. total_env: broadband RMS envelope of the raw window (same RMS
  machinery, no extra bandpass -- preprocess.py's upstream 0.5-40Hz filter
  already band-limits it, so a second broadband filter would be redundant).

  "Physiologically normalised" = classic relative/relative-power framing
  from clinical EEG analysis (a channel's delta fraction of total power,
  etc.) -- this is the actual normalisation mechanism, and it's the part
  that targets RC1 directly: relative power is invariant to the absolute
  amplitude scale that varies electrode-to-electrode and patient-to-
  patient (impedance, skull thickness, amplifier gain), which is exactly
  the "activation-statistics mismatch" RC1 describes for raw-EEG-fed CNNs.

BUG FOUND AND FIXED DURING SCOPING (disclosed, not swept under the rug)
--------------------------------------------------------------------------
First version used EPS=1e-8 (matching longctx_features.py's own ratio
formula) and an unclipped ratio channel. A synthetic sanity check
(near-zero-power/flat window, the kind that shows up for real on
artifact-heavy or lead-off segments) blew the ratio channel up to ~2e9 on
that one window -- because 1e-8 is negligible relative to the ~0-200
magnitude these [0,255]-domain envelopes actually take, it barely
regularises the division. A single such window would then dominate the
per-channel min-max fit and silently crush every other window's ratio
channel toward zero after scaling -- exactly the class of silent
numerical bug this project has been bitten by before (chb10 dual-scaler,
sign-flip). Fixed two ways, both required:
  1. EPS raised to 1.0 -- same physical units as the envelopes themselves
     (a [0,255]-domain "noise floor"), not an arbitrarily tiny constant.
  2. ch2 (ratio only -- rel_delta/rel_beta don't need this, they're
     naturally bounded near [0,1] by construction) clipped to
     [0, ratio_cap], ratio_cap = min(20.0, 99.9th percentile of the
     TRAIN split's raw ratio) -- fit on train, applied unchanged to
     val/test, same fit/apply discipline scale_channels_to_uint8() uses
     for the per-channel scaler itself. ratio_cap is recorded in the
     scaler JSON.

MEMORY BUG FOUND AND FIXED AFTER A REAL chb15 OOM (disclosed)
--------------------------------------------------------------------------
chb15 -- the largest of the six held-out patients, ~72K windows across
train/val/train_real/test -- got OOM-killed by an earlier version of this
script mid-run (after all envelope computation finished, before the final
save). Root cause: the final per-split arrays were stored as float32 and
held simultaneously in save_kwargs before the single savez_compressed
call, PLUS each split's stacking step briefly held both the three
separate float32 channel arrays and a newly-stacked float32 3-channel
array at once. Fixed by (1) storing the final output as uint8 -- values
are already clipped to [0,255], so this loses nothing meaningful, and
train_baseline.py/eval_event_level.py already upcast to float32 on load
regardless of on-disk dtype -- and (2) scale_channels_to_uint8() writing
directly into a preallocated uint8 buffer per channel instead of ever
materialising a stacked float32 intermediate. Measured on a chb15-scale
synthetic dataset before shipping this fix, not just reasoned about.

NOT the long-context design (Gate 2) -- deliberately distinct
------------------------------------------------------------------
Gate 2's rolling_band_ratio() computes a similar delta/beta ratio, but
over a long (12s) CAUSAL lookback, for temporal-context reasons. G
computes its ratio (and the two relative-power channels) over the CURRENT
window only, at the same granularity as the existing per-window envelope
machinery, because the interesting variable here is normalisation
(removing amplitude scale), not context length. Reusing the long-context
machinery directly would confound attribution -- a result would leave it
unclear whether G's normalisation was doing anything, or whether Gate 2's
already-tested context-length effect was just showing up again. Keeping
the two mechanisms separate matters doubly because Gate 2's long-context
channels are ALSO the confirmed-negative precedent for compounding a new
representation change onto C2 without a clean phase-1 result first
(Methodology Ledger) -- G needs to be a clean, independently-attributable
test of its own mechanism, not a repackaging of one already tried.

MEMORY DESIGN
-------------
Identical strategy to build_dataset_stft.py: build on top of the
already-split/SMOTE'd chbXX_dataset_ann.npz (six held-out eval patients)
or the already-pooled/split/per-domain-SMOTE'd multi_dataset_ann.npz (the
C1 training pool) -- never touch raw EDFs or full per-file window arrays.

DELIBERATE DEPARTURE from build_dataset_stft.py's own --multi-patient path
----------------------------------------------------------------------------
build_dataset_stft.py's build_multi_patient() re-pools from each patient's
raw per-patient _dataset_ann.npz using its own MULTI_PATIENTS constant
(which, on inspection, includes chb03 -- inconsistent with every other
candidate's chb01/02/05-only pool and with chb03's status as a held-out
eval patient everywhere else in this project). This script does NOT reuse
that path. Instead it builds G-features on top of the ALREADY-BUILT
multi_dataset_ann.npz (build_dataset_multi.py's real output: chb01/02/05,
proper Gate-0a real-pool split, proper per-domain SMOTE) so phase 1's
C1-anchored comparison is against the exact same pool windows A/C were
compared against -- only the channel transform differs. This is flagged
explicitly rather than silently diverging from precedent.

Usage
-----
  python3 src/preprocessing/build_dataset_g.py --patient chb03
  python3 src/preprocessing/build_dataset_g.py --patient chb10
  python3 src/preprocessing/build_dataset_g.py --patient chb13
  python3 src/preprocessing/build_dataset_g.py --patient chb15
  python3 src/preprocessing/build_dataset_g.py --patient chb16
  python3 src/preprocessing/build_dataset_g.py --patient chb20
  python3 src/preprocessing/build_dataset_g.py --multi-patient
      # requires data/processed/multi_dataset_ann.npz already built:
      #   python3 src/preprocessing/build_dataset_multi.py --patients chb01 chb02 chb05

Gate check (per patient / pool):
  python3 -c "
  import numpy as np
  d = np.load('data/processed/chb03_dataset_g.npz')
  print('X_train:', d['X_train'].shape)        # (N, 18, 512, 3)
  print('range:', d['X_train'].min(), d['X_train'].max())   # [0, 255]
  assert d['X_train'].shape[1:] == (18, 512, 3)
  print('Gate PASSED')
  "

Output files
------------
  data/processed/chbXX_dataset_g.npz    keys: X_train/y_train/X_val/y_val/X_test/y_test
                                         (+ X_train_real/y_train_real, carried through
                                         if present upstream -- not needed by phase 1's
                                         frozen-pool eval, kept for forward compatibility)
  data/processed/multi_dataset_g.npz    keys: X_train/y_train/X_val/y_val
                                         (+ domain_train/domain_val carried through
                                         unchanged if present -- NOT used by phase 1's
                                         plain supervised train, kept for forward
                                         compatibility only; no X_test, matches
                                         multi_dataset_ann.npz's own shape)
  data/processed/chbXX_scaler_g.json / multi_scaler_g.json   per-channel scale params
                                         + feature-definition metadata (bands, EPS,
                                         ratio_cap) for manifest/reproducibility
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, '.')
from src.preprocessing.build_dataset_stft import (
    butter_bandpass_sos, compute_band_envelope,
    DELTA_BAND, BETA_BAND, RMS_WIN_SAMPLES, CHUNK_SIZE,
)

EPS = 1.0          # scale-appropriate for [0,255]-domain envelopes -- see module docstring
RATIO_CAP_MAX = 20.0   # hard ceiling regardless of what the train-split percentile says


def compute_total_envelope(X: np.ndarray, label: str = '') -> np.ndarray:
    """
    Broadband RMS envelope of the raw window. Same chunked RMS machinery as
    compute_band_envelope(), minus the bandpass step -- the input is already
    band-limited to 0.5-40Hz by preprocess.py, so an extra broadband filter
    here would be redundant, not just wasted compute.
    """
    N = len(X)
    out = np.empty_like(X)
    n_chunks = (N + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci in range(n_chunks):
        s = ci * CHUNK_SIZE
        e = min(s + CHUNK_SIZE, N)
        sq = X[s:e] ** 2
        rms = uniform_filter1d(sq, size=RMS_WIN_SAMPLES, axis=-1, mode='reflect')
        np.sqrt(np.maximum(rms, 0.0), out=rms)
        out[s:e] = rms
        del sq, rms
        pct = 100.0 * e / N
        tag = f"[{label}] " if label else ""
        print(f"    {tag}total envelope  chunk {ci + 1}/{n_chunks}  {e}/{N} ({pct:.0f}%)",
              flush=True)
    return out


def compute_relpower_raw(X_raw: np.ndarray, label: str = ''):
    """
    Returns the three UNCLIPPED, UNSCALED channels plus the raw ratio
    (needed separately so the caller can fit ratio_cap on the train split
    only, then apply that same cap to val/test).

    Args:
        X_raw: (N, 18, 512) float32 -- raw EEG, already scaled to [0,255]
               (as stored in chbXX_dataset_ann.npz / multi_dataset_ann.npz)

    Returns:
        rel_delta, rel_beta, ratio_raw : each (N, 18, 512) float32
    """
    print(f"  Delta envelope ({DELTA_BAND[0]}-{DELTA_BAND[1]} Hz)...")
    delta_env = compute_band_envelope(X_raw, DELTA_BAND, label=label)
    print(f"  Beta envelope ({BETA_BAND[0]}-{BETA_BAND[1]} Hz)...")
    beta_env = compute_band_envelope(X_raw, BETA_BAND, label=label)
    print(f"  Total (broadband) envelope...")
    total_env = compute_total_envelope(X_raw, label=label)

    rel_delta = (delta_env / (total_env + EPS)).astype(np.float32)
    rel_beta = (beta_env / (total_env + EPS)).astype(np.float32)
    ratio_raw = (delta_env / (beta_env + EPS)).astype(np.float32)

    del delta_env, beta_env, total_env
    return rel_delta, rel_beta, ratio_raw


def scale_channels_to_uint8(rel_delta, rel_beta, ratio_raw, ratio_cap, scaler=None):
    """
    Combines clipping + (optionally fitting) + scaling + uint8 downcast into
    one pass, writing directly into a preallocated (N,18,512,3) uint8 output
    -- NEVER materialises a full float32 3-channel array. Each channel's
    float32 source is deleted immediately after it's written.

    This replaces an earlier two-step version (_finish() stacking a float32
    3ch array via np.stack, then a separate apply_scaler() call) that still
    peaked at ~6x a single channel's size even after the uint8-output fix,
    because the stack step itself briefly held both the three separate
    float32 channel arrays AND the newly-stacked float32 array at once.
    Found via a real peak-RSS measurement on a chb15-scale synthetic
    dataset (chb15 -- 6 held-out patients' largest, ~72K windows across
    train/val/train_real/test -- was the one that actually OOM-killed):
    the uint8-output-only fix reduced measured peak RSS by ~23% at 1/10
    scale, not the ~75% the byte-math alone suggested, because the working
    memory during the stack step (not just the final stored size) was the
    part still dominating. This version removes that stack step entirely.

    Args:
        rel_delta, rel_beta, ratio_raw: each (N, 18, 512) float32, UNCLIPPED
        ratio_cap: float, pre-computed on the train split's ratio_raw
        scaler: None (fits a new per-channel min/max scaler from these three
                arrays -- the train call) or an existing scaler dict (applies
                it unchanged -- the val/train_real/test calls)

    Returns:
        (out_uint8, scaler_dict) if scaler was None, else out_uint8 only.
    """
    ratio = np.clip(ratio_raw, 0.0, ratio_cap)
    del ratio_raw
    channels = [rel_delta, rel_beta, ratio]
    N = rel_delta.shape[0]
    out = np.empty((N, 18, 512, 3), dtype=np.uint8)

    fit_mode = scaler is None
    if fit_mode:
        scaler = {}

    for ch in range(3):
        arr = channels[ch]
        if fit_mode:
            lo, hi = float(arr.min()), float(arr.max())
            scaler[f'ch{ch}_min'] = lo
            scaler[f'ch{ch}_max'] = hi
        else:
            lo, hi = scaler[f'ch{ch}_min'], scaler[f'ch{ch}_max']
        span = (hi - lo) if (hi - lo) > 1e-12 else 1.0
        scaled = np.clip((arr - lo) / span * 255.0, 0.0, 255.0)
        out[..., ch] = scaled.astype(np.uint8)
        del scaled
        channels[ch] = None   # drop reference to this channel's float32 source

    del rel_delta, rel_beta, ratio, channels
    if fit_mode:
        return out, scaler
    return out


def build_one(in_path: str, out_path: str, scaler_out_path: str, tag: str):
    print(f"\n{'=' * 60}")
    print(f"Candidate G (phase 1) -- {tag}")
    print(f"Source: {in_path}")
    print(f"{'=' * 60}")
    data = np.load(in_path)

    has_test = 'X_test' in data.files
    X_train = data['X_train'].astype('float32')
    X_val = data['X_val'].astype('float32')
    y_train = data['y_train']
    y_val = data['y_val']
    if X_train.shape[1:] != (18, 512):
        sys.exit(f"ERROR: expected (18,512) raw windows in {in_path}, got "
                 f"{X_train.shape[1:]} -- G phase 1 expects the standard "
                 "1-channel _dataset_ann.npz / multi_dataset_ann.npz layout "
                 "as input, not an already-multichannel dataset.")

    print(f"\nTrain: {len(X_train)} windows (seizure={int(y_train.sum())})")
    rd_tr, rb_tr, ratio_raw_tr = compute_relpower_raw(X_train, label='train')
    del X_train
    gc.collect()

    # ratio_cap fit on TRAIN ONLY, then applied unchanged to val/test/train_real
    # -- same fit-on-train/apply-elsewhere discipline as the per-channel scaler.
    ratio_cap = float(min(RATIO_CAP_MAX, np.percentile(ratio_raw_tr, 99.9)))
    print(f"\n[ratio_cap] fit on train split: {ratio_cap:.4f} "
          f"(min(RATIO_CAP_MAX={RATIO_CAP_MAX}, 99.9th percentile))")

    X_train_s, scaler = scale_channels_to_uint8(rd_tr, rb_tr, ratio_raw_tr, ratio_cap)
    del rd_tr, rb_tr, ratio_raw_tr
    gc.collect()
    scaler['ratio_cap'] = ratio_cap

    print(f"\nVal: {len(X_val)} windows (seizure={int(y_val.sum())})")
    rd_vl, rb_vl, ratio_raw_vl = compute_relpower_raw(X_val, label='val')
    del X_val
    X_val_s = scale_channels_to_uint8(rd_vl, rb_vl, ratio_raw_vl, ratio_cap, scaler=scaler)
    del rd_vl, rb_vl, ratio_raw_vl
    gc.collect()

    save_kwargs = dict(X_train=X_train_s, y_train=y_train, X_val=X_val_s, y_val=y_val)

    if 'X_train_real' in data.files:
        X_train_real = data['X_train_real'].astype('float32')
        print(f"\nTrain (real, pre-SMOTE): {len(X_train_real)} windows")
        rd_r, rb_r, ratio_raw_r = compute_relpower_raw(X_train_real, label='train_real')
        del X_train_real
        save_kwargs['X_train_real'] = scale_channels_to_uint8(
            rd_r, rb_r, ratio_raw_r, ratio_cap, scaler=scaler)
        save_kwargs['y_train_real'] = data['y_train_real']
        del rd_r, rb_r, ratio_raw_r
        gc.collect()

    if has_test:
        X_test = data['X_test'].astype('float32')
        y_test = data['y_test']
        print(f"\nTest: {len(X_test)} windows (seizure={int(y_test.sum())})")
        rd_te, rb_te, ratio_raw_te = compute_relpower_raw(X_test, label='test')
        del X_test
        save_kwargs['X_test'] = scale_channels_to_uint8(
            rd_te, rb_te, ratio_raw_te, ratio_cap, scaler=scaler)
        save_kwargs['y_test'] = y_test
        del rd_te, rb_te, ratio_raw_te
        gc.collect()

    if 'domain_train' in data.files:
        save_kwargs['domain_train'] = data['domain_train']
    if 'domain_val' in data.files:
        save_kwargs['domain_val'] = data['domain_val']

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez_compressed(out_path, **save_kwargs)
    print(f"\nSaved: {out_path}")
    print(f"X_train: {X_train_s.shape}  range=[{X_train_s.min():.1f}, {X_train_s.max():.1f}]")

    with open(scaler_out_path, 'w') as f:
        json.dump({
            'feature_type': 'g_relpower',
            'channels': ['rel_delta_power', 'rel_beta_power', 'delta_beta_ratio'],
            'delta_band_hz': list(DELTA_BAND),
            'beta_band_hz': list(BETA_BAND),
            'rms_win_samples': RMS_WIN_SAMPLES,
            'eps': EPS,
            'ratio_cap': ratio_cap,
            'ratio_cap_max': RATIO_CAP_MAX,
            'per_channel_minmax': scaler,
        }, f, indent=2)
    print(f"Saved: {scaler_out_path}")

    assert X_train_s.shape[1:] == (18, 512, 3), f"Shape check failed: {X_train_s.shape}"
    assert 0 <= X_train_s.min() and X_train_s.max() <= 255.01, "Range check failed"
    print("Gate check PASSED (shape + range)")


def main():
    parser = argparse.ArgumentParser(description='Build Candidate G relative-band-power dataset.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--patient', type=str)
    group.add_argument('--multi-patient', action='store_true')
    args = parser.parse_args()

    if args.multi_patient:
        in_path = 'data/processed/multi_dataset_ann.npz'
        if not os.path.exists(in_path):
            sys.exit(f"ERROR: {in_path} not found.\n"
                     "Run: python3 src/preprocessing/build_dataset_multi.py "
                     "--patients chb01 chb02 chb05")
        build_one(in_path,
                 'data/processed/multi_dataset_g.npz',
                 'data/processed/multi_scaler_g.json',
                 tag='multi-patient pool (chb01/02/05, per multi_dataset_ann.npz)')
    else:
        in_path = f'data/processed/{args.patient}_dataset_ann.npz'
        if not os.path.exists(in_path):
            sys.exit(f"ERROR: {in_path} not found.\n"
                     f"Run: python3 src/preprocessing/build_dataset.py --patient {args.patient}")
        build_one(in_path,
                 f'data/processed/{args.patient}_dataset_g.npz',
                 f'data/processed/{args.patient}_scaler_g.json',
                 tag=args.patient)


if __name__ == '__main__':
    try:
        import scipy
        print(f"scipy {scipy.__version__} ✓")
    except ImportError:
        sys.exit("ERROR: pip install scipy --break-system-packages")
    main()
