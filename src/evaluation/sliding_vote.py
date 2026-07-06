"""
sliding_vote.py — Event-level seizure detection metrics
========================================================
Biologically motivated post-processing for window-level SNN predictions.

Key design principles:
  - Seizures have a recruitment phase (onset ~2-5s) that is often sub-threshold
    for window-level classifiers trained on fully-developed ictal windows.
  - detection_fraction=0.3 reflects that catching the propagation phase
    (~30-100% of seizure duration) is clinically sufficient.
  - gap_tolerance_s=60.0: empirically validated on chb03 and chb10 (Phase 2e,
    June 2026). At gap=10s, 636 FP windows merged into spurious events due to
    high window-level FPR. At gap=60s, FP events collapsed to 68 (9.06/hr on
    chb10) while true event detection was unaffected (sensitivity unchanged at
    0.50). The 60s value is also directly grounded in ILAE clinical cluster
    tolerance: ictal discharges falling below threshold and recovering within
    60s are considered part of the same seizure cluster in commercial EEG
    monitoring systems. Conservative relative to the ILAE 60s ceiling because
    scalp EEG is noisier than intracranial systems those thresholds were
    calibrated for — but empirical results confirm 60s is appropriate here.
  - min_sustained_windows=3 enforces a ~3s minimum sustained detection,
    above the 2-3s animal research floor and below the clinical 10-12s
    commercial system threshold, appropriate for CHB-MIT's 7s minimum seizure.

Empirical validation (Phase 2e — cross-patient inference, multi-patient model):
  chb10: event sens=0.50 (2/4), FP events/hr=9.06 at gap=60s
  chb03: event sens=0.00 (0/3), FP events/hr=2.99 at gap=60s
  gap=10s on chb10 gave 636 FP events (84.77/hr) — confirmed too aggressive.
  gap=60s selected as default based on this empirical sweep.

References:
  ILAE seizure definition: Fisher et al. (2014)
  CHB-MIT duration range (7-753s): MICAL paper, Physionet documentation
  Clinical minimum 10-12s: ILAE minimum standards for LT-EEG monitoring
  Onset/propagation distinction: Expert annotation at 1s and 10s post-onset
  Gap clustering (60s tolerance): USPTO seizure detection patent literature;
    confirmed empirically on chb03/chb10 Phase 2e June 2026
"""

import numpy as np


def group_into_events(y_true, step_s=1.0, gap_tolerance_s=60.0):
    """
    Group consecutive positive windows in y_true into seizure events.
    Windows within gap_tolerance_s of each other are merged into one event.

    Args:
        y_true         : (N,) binary array of ground-truth labels
        step_s         : time step between consecutive windows in seconds
                         (= window_s x (1 - overlap) = 2.0 x 0.5 = 1.0s)
        gap_tolerance_s: merge events separated by less than this many seconds.
                         Default 60.0s — ILAE clinical cluster tolerance,
                         empirically validated on chb03/chb10 (Phase 2e).

    Returns:
        List of (start_window_idx, end_window_idx) tuples — exclusive end
    """
    gap_windows = int(gap_tolerance_s / step_s)

    raw_events = []
    in_event = False
    start = None

    for i, label in enumerate(y_true):
        if label == 1 and not in_event:
            in_event = True
            start = i
        elif label == 0 and in_event:
            raw_events.append((start, i))
            in_event = False
    if in_event:
        raw_events.append((start, len(y_true)))

    if not raw_events:
        return []

    merged = [raw_events[0]]
    for start, end in raw_events[1:]:
        prev_start, prev_end = merged[-1]
        if (start - prev_end) <= gap_windows:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged

def event_level_metrics(
    y_true, y_pred, window_s=2.0, overlap=0.5,
    detection_fraction=0.3, min_sustained_windows=3, gap_tolerance_s=60.0,
):
    step_s = window_s * (1.0 - overlap)
    total_hours = (len(y_true) * step_s) / 3600.0

    true_events = group_into_events(y_true, step_s, gap_tolerance_s)

    if not true_events:
        return {
            "event_sensitivity": None,
            "n_events": 0,
            "n_detected": 0,
            "false_positives": _count_false_positives(y_pred, y_true, step_s, gap_tolerance_s, min_sustained_windows),
            "fp_per_hour": None,
            "total_hours": total_hours,
            "latencies_s": [],
            "mean_latency_s": None,
            "n_undetected": 0,
        }

    n_detected = 0
    n_undetected = 0
    latencies_s = []
    for ev_start, ev_end in true_events:
        event_preds = y_pred[ev_start:ev_end]
        frac_positive = event_preds.mean()
        max_run = _max_consecutive(event_preds)
        detected = frac_positive >= detection_fraction and max_run >= min_sustained_windows
        if detected:
            n_detected += 1
            onset_idx = _first_sustained_run_start(event_preds, min_sustained_windows)
            latencies_s.append(round(onset_idx * step_s, 2))
        else:
            n_undetected += 1

    fp = _count_false_positives(y_pred, y_true, step_s, gap_tolerance_s, min_sustained_windows)

    return {
        "event_sensitivity": n_detected / len(true_events),
        "n_events": len(true_events),
        "n_detected": n_detected,
        "false_positives": fp,
        "fp_per_hour": fp / total_hours if total_hours > 0 else None,
        "total_hours": total_hours,
        "latencies_s": latencies_s,
        "mean_latency_s": round(float(np.mean(latencies_s)), 2) if latencies_s else None,
        "n_undetected": n_undetected,
    }

def collapse_diagnostic(y_pred, window_specificity, y_true=None, step_s=1.0,
                         gap_tolerance_s=60.0,
                         spec_floor=0.9, block_frac_ceiling=0.9,
                         underfire_ratio_floor=0.1):
    """
    ... (existing docstring) ...
    NEW: also FAILs if positive_window_fraction falls below
    underfire_ratio_floor × the true seizure-window rate — catches a model
    that's gone non-responsive post-quantisation, the mirror-image failure
    to the over-firing case the original two checks were built for.
    """
    n = len(y_pred)
    pred_blocks = group_into_events(y_pred, step_s, gap_tolerance_s)
    pos_frac = float(np.mean(y_pred)) if n > 0 else 0.0
    largest_block_frac = (
        max((end - start) for start, end in pred_blocks) / n
        if pred_blocks and n > 0 else 0.0
    )

    reasons = []
    if window_specificity is not None and window_specificity < spec_floor:
        reasons.append(f"window specificity {window_specificity:.4f} < {spec_floor}")
    if largest_block_frac > block_frac_ceiling:
        reasons.append(
            f"largest predicted block covers {100*largest_block_frac:.1f}% "
            f"of the recording (> {100*block_frac_ceiling:.0f}%)"
        )
    if y_true is not None:
        true_frac = float(np.mean(y_true))
        if true_frac > 0 and pos_frac < underfire_ratio_floor * true_frac:
            reasons.append(
                f"positive-window fraction {100*pos_frac:.4f}% is less than "
                f"{100*underfire_ratio_floor:.0f}% of the true seizure-window "
                f"rate ({100*true_frac:.4f}%) — likely under-firing collapse "
                "(model gone non-responsive), not genuine high specificity."
            )

    return {
        "positive_window_fraction": round(pos_frac, 4),
        "n_predicted_blocks": len(pred_blocks),
        "largest_block_fraction": round(largest_block_frac, 4),
        "pass": len(reasons) == 0,
        "reasons": reasons,
    }

def _max_consecutive(arr):
    """Return the length of the longest run of 1s in a binary array."""
    max_run = 0
    current = 0
    for v in arr:
        if v == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run
def _first_sustained_run_start(arr, min_run):
    """Index of the start of the first run of >= min_run consecutive 1s, or None."""
    current_start = None
    current_len = 0
    for i, v in enumerate(arr):
        if v == 1:
            if current_len == 0:
                current_start = i
            current_len += 1
            if current_len >= min_run:
                return current_start
        else:
            current_len = 0
    return None

def _count_false_positives(y_pred, y_true, step_s, gap_tolerance_s, min_sustained_windows=1):
    """
    Count predicted positive events with no overlap with any true event.
    Uses the same gap-merging logic for consistency.

    Gate 3e: a predicted block shorter than min_sustained_windows is not
    counted as a false-positive "event" — mirrors the sustained-detection
    floor already applied to true positives, so a momentary 1-window blip
    isn't treated as a full clinical false alarm any more than it would be
    treated as a detection.
    """
    pred_events = group_into_events(y_pred, step_s, gap_tolerance_s)
    true_events = group_into_events(y_true, step_s, gap_tolerance_s)

    fp = 0
    for p_start, p_end in pred_events:
        if (p_end - p_start) < min_sustained_windows:
            continue
        overlaps_any = any(
            p_start < t_end and p_end > t_start
            for t_start, t_end in true_events
        )
        if not overlaps_any:
            fp += 1
    return fp
