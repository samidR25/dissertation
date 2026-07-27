"""
audit_test_seizures.py
=======================
Cheap pre-download triage: estimates how many seizures would fall in the
chronological TEST split for a list of candidate patients, using only the
tiny chbNN-summary.txt files (no EDF download needed).

This mirrors preprocess.py's parse_chbmit_summary() exactly, so seizure
counts agree with what the real pipeline will see -- but the split-point
estimate is file-count-based, not duration/window-based, since summary
files alone don't give exact per-file durations. Treat the output as a
TRIAGE signal (download this patient or not?), not a final answer --
confirm with the real build_dataset.py gate-check snippet (also below)
once a patient is actually preprocessed.

Usage:
    python3 audit_test_seizures.py chb08 chb12 chb13 chb14 chb15 chb16 \
        chb17 chb18 chb19 chb22 chb23 chb24

Assumes each patient's summary file already exists at:
    data/raw/chbmit/physionet.org/files/chbmit/1.0.0/{patient}/{patient}-summary.txt
"""
import re
import sys
import os

DATA_ROOT = 'data/raw/chbmit/physionet.org/files/chbmit/1.0.0'


def parse_chbmit_summary(summary_path):
    """Identical logic to preprocess.py's parser -- kept in lockstep
    deliberately so triage numbers and real pipeline numbers never disagree."""
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
                end_line = lines[i].strip()
                start_s = int(re.search(r'(\d+)\s+second', start_line).group(1))
                end_s = int(re.search(r'(\d+)\s+second', end_line).group(1))
                seizures[current_file].append((start_s, end_s))
        i += 1
    return seizures


def estimate_test_split_seizures(patient, train_frac=0.70, val_frac=0.15):
    """
    Files are sorted by name (CHB-MIT's own numbering is chronological,
    gaps in numbering are fine -- chb01 has exactly this property and the
    real pipeline already relies on sorted-filename order elsewhere).

    APPROXIMATION: splits by file count, not by window/duration count.
    Most CHB-MIT files are ~1hr, but lengths do vary per patient (the
    project's own chb01 notes document files as short as 10 min) -- so
    this can be off by a file or two at the boundary. Good enough to
    decide "worth downloading?", not good enough to cite in the write-up.
    """
    summary_path = f'{DATA_ROOT}/{patient}/{patient}-summary.txt'
    if not os.path.exists(summary_path):
        return None, f"summary file not found at {summary_path}"

    seizures = parse_chbmit_summary(summary_path)
    files = sorted(seizures.keys())
    n = len(files)
    if n == 0:
        return None, "no files parsed -- check summary format"

    test_start_idx = int(n * (train_frac + val_frac))
    test_files = files[test_start_idx:]

    total_seizures = sum(len(v) for v in seizures.values())
    test_seizures = sum(len(seizures[f]) for f in test_files)
    test_files_with_seizures = [f for f in test_files if seizures[f]]

    return {
        'total_files': n,
        'total_seizures': total_seizures,
        'estimated_test_seizures': test_seizures,
        'test_files_with_seizures': test_files_with_seizures,
    }, None


if __name__ == '__main__':
    patients = sys.argv[1:]
    if not patients:
        sys.exit("Usage: python3 audit_test_seizures.py chb08 chb12 ...")

    print(f"{'Patient':<8} {'TotalSz':>8} {'EstTestSz':>10}  Verdict")
    print("-" * 60)
    for p in patients:
        result, err = estimate_test_split_seizures(p)
        if err:
            print(f"{p:<8} {'--':>8} {'--':>10}  SKIP ({err})")
            continue
        verdict = "PRIORITY" if result['estimated_test_seizures'] >= 2 else (
            "fragile (1)" if result['estimated_test_seizures'] == 1 else
            "SKIP (0 test seizures)")
        print(f"{p:<8} {result['total_seizures']:>8} "
              f"{result['estimated_test_seizures']:>10}  {verdict}")
        if result['test_files_with_seizures']:
            print(f"         test files with seizures: {result['test_files_with_seizures']}")
