#!/usr/bin/env python3
"""
fix_coral_post_training_measurement.py
=========================================
Second CORAL hotfix. The first CORAL run's printed verdict ("NOT reduced --
investigate...") was itself a diagnostic bug, not a real finding: it compared
coral_monitor.history[0] (pre-training) against coral_monitor.history[-1]
(the LIVE weights at the end of the last epoch trained). But
EarlyStopping(restore_best_weights=True) restores the model to its best
val_output_loss epoch's weights (here, epoch 11) AFTER that last on_epoch_end
fires -- so history[-1] reflects a later, more-overfit epoch than the
checkpoint that was actually saved, converted, and evaluated. The eval
numbers (sens/spec/FPR-hr) were unaffected -- those already came from the
correctly-restored ModelCheckpoint file -- only the covariance-distance
verdict was measuring the wrong model state.

Fix: measure the "post-training" covariance-distance directly against
best_coral (loaded from the ModelCheckpoint .h5 file, right after fit()
returns) instead of trusting anything from mid-training callback ordering.
This is unambiguously the same checkpoint that gets extracted, converted,
and deployed -- no dependency on which callback's on_train_end fires first.

Run from ~/dissertation/ with akida_env activated, AFTER
fix_coral_fstring_syntax.py has already been applied:
    python3 fix_coral_post_training_measurement.py

Hard-refuses if either anchor isn't found exactly once.
"""
import sys

PATH = 'src/models/train_baseline.py'

old_1 = """    from src.models.akida_cnn_v2 import (build_seizure_cnn_v2_coral,
                                          extract_deployable_submodel,
                                          make_coral_loss,
                                          CoralDistanceMonitor)"""

new_1 = """    from src.models.akida_cnn_v2 import (build_seizure_cnn_v2_coral,
                                          extract_deployable_submodel,
                                          make_coral_loss,
                                          coral_pairwise_distance,
                                          CoralDistanceMonitor)"""

old_2 = """    pre_d  = coral_monitor.history[0][1]
    post_d = coral_monitor.history[-1][1]
    _coral_verdict = ('reduced' if post_d < pre_d
                       else 'NOT reduced -- investigate before trusting this run')
    print(f"\\n[CORAL] Covariance-distance readout, pre-training -> "
          f"post-training: {pre_d:.6f} -> {post_d:.6f}  ({_coral_verdict})")"""

new_2 = """    # Measure against the ACTUAL restored/deployed checkpoint (best_coral,
    # loaded from disk post-fit()), not coral_monitor.history[-1] -- that
    # reflected live end-of-training weights, which EarlyStopping's
    # restore_best_weights may have since superseded with an earlier,
    # better-val-loss epoch. No dependency on callback firing order.
    _, best_feats = best_coral.predict(X_val, verbose=0, batch_size=64)
    pre_d  = coral_monitor.history[0][1]
    post_d = coral_pairwise_distance(best_feats, domain_val, n_domains)
    _coral_verdict = ('reduced' if post_d < pre_d
                       else 'NOT reduced -- investigate before trusting this run')
    print(f"\\n[CORAL] Covariance-distance readout, pre-training -> "
          f"post-training (restored/deployed checkpoint): "
          f"{pre_d:.6f} -> {post_d:.6f}  ({_coral_verdict})")"""

with open(PATH, 'r') as f:
    content = f.read()

for i, (old, new) in enumerate([(old_1, new_1), (old_2, new_2)]):
    n = content.count(old)
    if n == 0:
        sys.exit(f"REFUSING: anchor #{i} not found in {PATH} -- either "
                  "already fixed, or the file doesn't match what this "
                  "hotfix expects. No changes written.")
    if n > 1:
        sys.exit(f"REFUSING: anchor #{i} matches {n} times (expected 1). "
                  "No changes written.")
    content = content.replace(old, new)

with open(PATH, 'w') as f:
    f.write(content)

print(f"Patched: {PATH}")
print("Re-run: python3 -m py_compile src/models/train_baseline.py")
