#!/usr/bin/env python3
"""
compare_weights.py
===================
Numerically compares layer weights between two .h5 checkpoints, since md5
hashing an .h5 file tests byte-identity of the SERIALIZED FILE, not the
underlying weight tensors -- a model loaded and re-saved via model.save()
can produce a different file hash despite carrying identical weights
(different HDF5 metadata/chunking). This resolves whether "same hash" or
"different hash" actually means "same weights" or "different weights" for
this project's per-patient checkpoint-copy convention.

Run from ~/dissertation/ with akida_env activated:
    python3 compare_weights.py <model_a.h5> <model_b.h5>

Prints per-layer max-abs-difference and an overall verdict. Uses
np.allclose with a small tolerance (float32 round-trip noise), not exact
equality, since re-saving through HDF5 can introduce tiny float rounding
even for genuinely identical weights.
"""
import sys
import numpy as np
import tf_keras as keras

if len(sys.argv) != 3:
    sys.exit("Usage: python3 compare_weights.py <model_a.h5> <model_b.h5>")

path_a, path_b = sys.argv[1], sys.argv[2]

print(f"Loading {path_a} ...")
model_a = keras.models.load_model(path_a, compile=False)
print(f"Loading {path_b} ...")
model_b = keras.models.load_model(path_b, compile=False)

weights_a = model_a.get_weights()
weights_b = model_b.get_weights()

if len(weights_a) != len(weights_b):
    sys.exit(f"REFUSING to compare: different number of weight arrays "
              f"({len(weights_a)} vs {len(weights_b)}) -- these are "
              "structurally different models, not just different weight "
              "values. No further comparison is meaningful.")

print(f"\n{'Layer idx':>10}  {'Shape':>20}  {'Max |diff|':>12}  {'Identical?':>10}")
print("-" * 60)
all_identical = True
for i, (wa, wb) in enumerate(zip(weights_a, weights_b)):
    if wa.shape != wb.shape:
        print(f"{i:>10}  SHAPE MISMATCH: {wa.shape} vs {wb.shape}")
        all_identical = False
        continue
    diff = float(np.max(np.abs(wa.astype('float64') - wb.astype('float64'))))
    identical = np.allclose(wa, wb, atol=1e-6, rtol=1e-5)
    if not identical:
        all_identical = False
    print(f"{i:>10}  {str(wa.shape):>20}  {diff:>12.8f}  {str(identical):>10}")

print("\n" + "=" * 60)
if all_identical:
    print(f"VERDICT: {path_a} and {path_b} have IDENTICAL weights "
          "(within float32 tolerance). Any file-hash difference between "
          "them is a serialization artifact, not a real weight "
          "difference -- these are the SAME model.")
else:
    print(f"VERDICT: {path_a} and {path_b} have DIFFERENT weights. "
          "This is a genuine weight divergence, not a serialization "
          "artifact -- these are DIFFERENT trained models.")
