"""
cleanup_redundant_raw_windows.py
==================================
Safely reclaim disk space by deleting per-patient raw windowed arrays
({patient}_X.npy / {patient}_y.npy) ONLY for patients that already have a
confirmed dataset_ann.npz with a valid, matching manifest. Mirrors the
chb18/chb19 reasoning already confirmed safe this session: nothing
downstream reads _X.npy/_y.npy again once dataset_ann.npz exists, except
run_patient.py's skip-check (which just re-runs preprocess.py if missing —
a few minutes of recomputation, not a correctness issue, since the raw
EDFs would need to still be on disk for that anyway).

Refuses to delete a patient's raw windows if:
  - the corresponding dataset_ann.npz doesn't exist, or
  - its manifest doesn't exist / doesn't load, or
  - the manifest's recorded patient field doesn't match the filename

Keeps {patient}_y.npy (tiny) for chb10 specifically, since Gate 1's
self-check script regression-tests against it. Everything else (X.npy
and y.npy) is deleted for patients that pass the check.

Usage:
    python3 cleanup_redundant_raw_windows.py            # dry run (default)
    python3 cleanup_redundant_raw_windows.py --execute  # actually delete
"""
import argparse
import json
import os

DATA_DIR = 'data/processed'
KEEP_Y_FOR = {'chb10'}  # Gate 1 regression check depends on chb10_y.npy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete. Without this, just reports '
                             'what WOULD be deleted and how much space '
                             'would be reclaimed.')
    args = parser.parse_args()

    total_bytes = 0
    candidates = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith('_X.npy'):
            continue
        patient = fname[:-len('_X.npy')]
        x_path = os.path.join(DATA_DIR, fname)
        y_path = os.path.join(DATA_DIR, f'{patient}_y.npy')
        ann_path = os.path.join(DATA_DIR, f'{patient}_dataset_ann.npz')
        manifest_path = ann_path + '.manifest.json'

        if not os.path.exists(ann_path):
            print(f"SKIP {patient}: no {ann_path} — keeping raw windows")
            continue
        if not os.path.exists(manifest_path):
            print(f"SKIP {patient}: no manifest for {ann_path} — keeping "
                  "raw windows (can't confirm provenance)")
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get('patient') != patient:
            print(f"SKIP {patient}: manifest patient field "
                  f"'{manifest.get('patient')}' != '{patient}' — refusing "
                  "to guess, keeping raw windows")
            continue

        keep_y = patient in KEEP_Y_FOR
        size = os.path.getsize(x_path)
        if not keep_y and os.path.exists(y_path):
            size += os.path.getsize(y_path)
        total_bytes += size
        candidates.append((patient, x_path, None if keep_y else y_path, size))
        keep_note = " (keeping y.npy for Gate 1 regression check)" if keep_y else ""
        print(f"OK   {patient}: confirmed dataset_ann.npz + manifest match — "
              f"{size/1024**3:.2f}GiB reclaimable{keep_note}")

    print(f"\nTotal reclaimable: {total_bytes/1024**3:.2f} GiB across "
          f"{len(candidates)} patients")

    if not args.execute:
        print("\nDRY RUN — nothing deleted. Re-run with --execute to apply.")
        return

    for patient, x_path, y_path, size in candidates:
        os.remove(x_path)
        if y_path and os.path.exists(y_path):
            os.remove(y_path)
        print(f"Deleted {patient} raw windows ({size/1024**3:.2f}GiB)")

    print(f"\nReclaimed {total_bytes/1024**3:.2f} GiB.")


if __name__ == '__main__':
    main()
