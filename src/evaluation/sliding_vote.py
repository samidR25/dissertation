"""
sliding_vote.py — Event-level seizure detection metrics
========================================================
Biologically motivated post-processing for window-level SNN predictions.

Key design principles:
  - Seizures have a recruitment phase (onset ~2-5s) that is often sub-threshold
    for window-level classifiers trained on fully-developed ictal windows.
  - Detection_fraction=0.3 reflects that catching the propagation phase
    (~30-100% of seizure duration) is clinically sufficient.
  - Gap tolerance bridges brief sub-threshold dips within a seizure,
    consistent with ILAE clustering criteria.
  - Minimum sustained windows enforces the clinical ~10s floor operationally,
    reduced to ~3s (3 windows) to accommodate the 7-753s range in CHB-MIT.

References:
  ILAE seizure definition: Fisher et al. (2014)
  CHB-MIT duration range (7-753s): MICAL paper, Physionet documentation
  Clinical minimum 10-12s: ILAE minimum standards for LT-EEG monitoring
  Onset/propagation distinction: Expert annotation at 1s and 10s post-onset
  Gap clustering (60s tolerance): USPTO seizure detection patent literature
"""

import numpy as np


def group_into_events(y_true, step_s=1.0, gap_tolerance_s=10.0):
    """
    Group consecutive positive windows in y_true into seizure events.
    Windows within gap_tolerance_s of each other are merged into one event.

    Args:
        y_true         : (N,) binary array of ground-truth labels
        step_s         : time step between consecutive windows in seconds
                         (= window_s × (1 - overlap) = 2.0 × 0.5 = 1.0s)
        gap_tolerance_s: merge events separated by less than this many seconds
                         Biological basis: ictal discharges can briefly dip
                         below threshold mid-seizure; 10s is conservative
                         relative to ILAE's 60s clinical cluster tolerance.

    Returns:
        List of (start_window_idx, end_window_idx) tuples — exclusive end
    """
    gap_windows = int(gap_tolerance_s / step_s)

    # Find raw ictal runs (contiguous blocks of y==1)
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

    # Merge events separated by less than gap_tolerance
    merged = [raw_events[0]]
    for start, end in raw_events[1:]:
        prev_start, prev_end = merged[-1]
        if (start - prev_end) <= gap_windows:
            merged[-1] = (prev_start, end)  # extend previous event
        else:
            merged.append((start, end))

    return merged


def event_level_metrics(
    y_true,
    y_pred,
    window_s=2.0,
    overlap=0.5,
    detection_fraction=0.3,
    min_sustained_windows=3,
    gap_tolerance_s=10.0,
):
    """
    Compute event-level sensitivity and false positive rate.

    An event is DETECTED if:
      (a) >= detection_fraction of its windows are predicted positive, AND
      (b) at least min_sustained_windows consecutive positives exist within it.

    Condition (b) implements the clinical ~10s floor (3 windows × 1s step = 3s
    minimum sustained detection) while remaining sensitive to short seizures
    in CHB-MIT (minimum duration 7s = ~6 windows at 50% overlap).

    Args:
        y_true              : (N,) ground-truth binary labels
        y_pred              : (N,) predicted binary labels
        window_s            : window duration in seconds (default 2.0)
        overlap             : fractional window overlap (default 0.5)
        detection_fraction  : fraction of event windows that must be positive
                              (default 0.3 — accommodates undetectable onset phase)
        min_sustained_windows: minimum consecutive positive windows required
                              (default 3 ≈ 3s at 1s step — above animal floor,
                               below clinical 10-12s commercial system threshold)
        gap_tolerance_s     : seconds of sub-threshold signal tolerated within
                              an event before it is considered terminated
                              (default 10s — conservative vs ILAE 60s)

    Returns:
        dict with keys:
            event_sensitivity : float or None (None if no events in y_true)
            n_events          : int
            n_detected        : int
            false_positives   : int   (predicted events with no true event overlap)
            fp_per_hour       : float
            total_hours       : float
    """
    step_s = window_s * (1.0 - overlap)  # = 1.0s for our 2s/50% config
    total_hours = (len(y_true) * step_s) / 3600.0

    true_events = group_into_events(y_true, step_s, gap_tolerance_s)

    if not true_events:
        return {
            "event_sensitivity": None,
            "n_events": 0,
            "n_detected": 0,
            "false_positives": _count_false_positives(y_pred, y_true, step_s, gap_tolerance_s),
            "fp_per_hour": None,
            "total_hours": total_hours,
        }

    n_detected = 0
    for ev_start, ev_end in true_events:
        event_preds = y_pred[ev_start:ev_end]
        n_windows = ev_end - ev_start

        # Criterion (a): fraction of windows positive
        frac_positive = event_preds.mean()

        # Criterion (b): longest consecutive run of positives
        max_run = _max_consecutive(event_preds)

        if frac_positive >= detection_fraction and max_run >= min_sustained_windows:
            n_detected += 1

    fp = _count_false_positives(y_pred, y_true, step_s, gap_tolerance_s)

    return {
        "event_sensitivity": n_detected / len(true_events),
        "n_events": len(true_events),
        "n_detected": n_detected,
        "false_positives": fp,
        "fp_per_hour": fp / total_hours if total_hours > 0 else None,
        "total_hours": total_hours,
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


def _count_false_positives(y_pred, y_true, step_s, gap_tolerance_s):
    """
    Count predicted positive events that have no overlap with any true event.
    Uses the same gap-merging logic for consistency.
    """
    pred_events = group_into_events(y_pred, step_s, gap_tolerance_s)
    true_events = group_into_events(y_true, step_s, gap_tolerance_s)

    fp = 0
    for p_start, p_end in pred_events:
        overlaps_any = any(
            p_start < t_end and p_end > t_start
            for t_start, t_end in true_events
        )
        if not overlaps_any:
            fp += 1
    return fp
