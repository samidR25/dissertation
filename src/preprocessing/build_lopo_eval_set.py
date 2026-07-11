"""
src/preprocessing/build_lopo_eval_set.py  (v2 — fixes OOM kill)
==================================================================
Builds the FULL-recording evaluation set for one LOPO held-out patient.

v2 fix: the original version built X_full = np.array(X, ...) then computed
X_full_s = (X_full * scale + shift).clip(...).astype(...) as one expression.
Each operator in that chain (multiply, add, clip, astype) allocates its own
full-recording-sized temporary array, so peak memory was ~3-4x the size of
the final output. For chb10 (~240k windows, ~8.85GB per copy) that's ~30GB+
of transient arrays for a single patient -- got OOM-killed. Fixed by
processing in fixed-size chunks straight from the memmap into ONE
preallocated output buffer, in-place, so peak memory is ~1x the final
output size (+ one small chunk), not 3-4x.

Why full recording, not the existing chronological 15% test slice:
  Standard LOPO (and the Ali et al. 2024 comparator this project is being
  read against) evaluates the held-out patient on their ENTIRE recording,
  because that patient contributed exactly 0% of their data to training in
  this fold -- there is no "seen" portion to exclude.

Scaling: reuses the patient's OWN existing scaler
  (data/processed/<patient>_scaler.json, produced by the ORIGINAL
  per-patient build_dataset.py run). Data-normalisation step, not label
  information -- same justification already used elsewhere in this project
  (e.g. the chb06 dual-scaler discussion).

Usage:
    python3 src/preprocessing/build_lopo_eval_set.py --patient chb10
    python3 src/preprocessing/build_lopo_eval_set.py --patient chb06 --chunk-size 2000

Output:
    data/processed/<patient>_dataset_lopo_full.npz
        keys: X_test, y_test  (ALL windows, ALL labels, scaled [0,255])
"""
import argparse, json, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--patient', required=True)
parser.add_argument('--chunk-size', type=int, default=5000,
                    help="Windows processed per chunk during scaling. Bounds "
                         "peak RAM to roughly one chunk (a few hundred MB at "
                         "the default) plus the single output buffer -- not "
                         "multiple full-recording-sized temporaries. Lower "
                         "this if a patient's recording still OOMs even with "
                         "chunking (unlikely -- the output buffer itself is "
                         "the real floor, see note below).")
args = parser.parse_args()

X_path = f'data/processed/{args.patient}_X.npy'
y_path = f'data/processed/{args.patient}_y.npy'
scaler_path = f'data/processed/{args.patient}_scaler.json'

for p in (X_path, y_path, scaler_path):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found.\n"
            f"Run first: python3 src/preprocessing/preprocess.py --patient {args.patient}\n"
            f"           python3 src/preprocessing/build_dataset.py --patient {args.patient}"
            f"  (only needed once, to produce {scaler_path})")

X = np.load(X_path, mmap_mode='r')
y = np.array(np.load(y_path), dtype=np.int32)
with open(scaler_path) as f:
    scaler = json.load(f)

N = len(y)
n_seiz = int(y.sum())
est_gib = N * 18 * 512 * 4 / 1024**3
print(f"Patient       : {args.patient}")
print(f"Full recording: {N} windows (seizure={n_seiz}, {100*n_seiz/max(N,1):.3f}%)")
print(f"Scaler        : {scaler_path} (scale={scaler['scale']:.4f}, shift={scaler['shift']:.2f})")
print(f"Output buffer : ~{est_gib:.2f}GiB (unavoidable floor -- this is the "
      f"final array size, not a temporary; if THIS alone doesn't fit in "
      f"RAM, --chunk-size won't help and uint8 storage would be the next "
      f"lever, since the scaled range is exactly [0,255])")

if n_seiz == 0:
    raise ValueError(
        f"{args.patient} has zero seizure windows in its FULL recording -- "
        "cannot be a LOPO held-out fold. Check preprocessing.")

scale = float(scaler['scale'])
shift = float(scaler['shift'])

# Single pre-allocated output buffer -- the fix. Filled chunk-by-chunk
# in-place, straight from the memmap, so no other full-recording-sized
# array ever exists at the same time as this one.
X_full_s = np.empty((N, 18, 512), dtype='float32')
chunk = args.chunk_size
for start in range(0, N, chunk):
    end = min(start + chunk, N)
    buf = np.array(X[start:end], dtype='float32')      # always copies (unlike asarray)
    buf *= scale
    buf += shift
    np.clip(buf, 0, 255, out=buf)
    X_full_s[start:end] = buf
    if (start // chunk) % 10 == 0:
        print(f"  scaled {end}/{N} windows...")

y_full = y

os.makedirs('data/processed', exist_ok=True)
out_path = f'data/processed/{args.patient}_dataset_lopo_full.npz'
np.savez_compressed(out_path, X_test=X_full_s, y_test=y_full)

print(f"\nSaved  : {out_path}")
print(f"X_test : {X_full_s.shape}  range=[{X_full_s.min():.1f}, {X_full_s.max():.1f}]")
print(f"y_test : seizure={int(y_full.sum())} ({100*y_full.mean():.3f}%)")
print("\n✓ Sanity checks passed")
