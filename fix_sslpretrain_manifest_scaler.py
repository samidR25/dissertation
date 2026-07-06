#!/usr/bin/env python3
"""
fix_sslpretrain_manifest_scaler.py
=====================================
_write_ckpt_manifest()'s scaler-path resolution only recognised
patient_tag == 'multi' or patient_tag.startswith('multi_from_') as "use
the pool scaler" -- missed the new '_sslpretrain' suffix from Candidate C
(apply_ssl_pretrain_init_patch.py), so results/best_ann_multi_sslpretrain_
v2.h5's manifest was written with no scaler provenance (harmless to the
already-completed training run -- the data itself was correctly scaled
via multi_dataset_ann.npz throughout -- but it weakens eval_event_level.
py's Gate 1b consistency check, which will warn and skip verification
instead of actively confirming).

Fix: extend both scaler-path branches (plain and _longctx_w) to also
match patient_tag.startswith('multi_sslpretrain').

Run from ~/dissertation/ with akida_env activated:
    python3 fix_sslpretrain_manifest_scaler.py

Then regenerate the manifest for the checkpoint already trained (no
retraining needed -- just rewrites the manifest):
    python3 -c "
import sys; sys.path.insert(0, '.')
from src.manifest import write_manifest
import json
with open('data/processed/multi_scaler.json') as f:
    scaler = json.load(f)
write_manifest('results/best_ann_multi_sslpretrain_v2.h5',
    patient_tag='multi_sslpretrain', seed=123,
    scaler_path='data/processed/multi_scaler.json', scaler=scaler,
    finetune_from=None, gradual_unfreeze=False, model_version=2)
"
"""
import sys

PATH = 'src/models/train_baseline.py'

old = """    if '_longctx_w' in patient_tag:
        _base_tag, _ws = patient_tag.split('_longctx_w')
        scaler_path = (f'data/processed/multi_scaler_longctx_w{_ws}.json'
                       if _base_tag == 'multi' or _base_tag.startswith('multi_from_')
                       else f'data/processed/{_base_tag}_scaler_longctx_w{_ws}.json')
    else:
        scaler_path = ('data/processed/multi_scaler.json'
                       if patient_tag == 'multi' or patient_tag.startswith('multi_from_')
                       else f'data/processed/{patient_tag}_scaler.json')"""

new = """    if '_longctx_w' in patient_tag:
        _base_tag, _ws = patient_tag.split('_longctx_w')
        scaler_path = (f'data/processed/multi_scaler_longctx_w{_ws}.json'
                       if _base_tag == 'multi' or _base_tag.startswith('multi_from_')
                       or _base_tag.startswith('multi_sslpretrain')
                       else f'data/processed/{_base_tag}_scaler_longctx_w{_ws}.json')
    else:
        scaler_path = ('data/processed/multi_scaler.json'
                       if patient_tag == 'multi' or patient_tag.startswith('multi_from_')
                       or patient_tag.startswith('multi_sslpretrain')
                       else f'data/processed/{patient_tag}_scaler.json')"""

with open(PATH, 'r') as f:
    content = f.read()

n = content.count(old)
if n == 0:
    sys.exit(f"REFUSING: anchor not found in {PATH}. No changes written.")
if n > 1:
    sys.exit(f"REFUSING: anchor matches {n} times (expected 1). No changes written.")

content = content.replace(old, new)
with open(PATH, 'w') as f:
    f.write(content)

print(f"Patched: {PATH}")
print("Re-run: python3 -m py_compile src/models/train_baseline.py")
