#!/usr/bin/env python3
"""
fix_coral_fstring_syntax.py
=============================
Hotfix for apply_coral_first_experiment_patch.py's one bug: a string
literal split across two physical lines inside an f-string's {} braces.
This is accepted by Python 3.12 (PEP 701 relaxed the tokenizer) but is a
SyntaxError on Python 3.11, which is what akida_env actually runs -- my
own syntax check ran on the wrong Python version and missed this.

Replaces the ternary-inside-f-string with a plain variable computed
first, then referenced inside a single-line f-string. No behaviour
change -- same "reduced" / "NOT reduced -- investigate..." message.

Run from ~/dissertation/ with akida_env activated:
    python3 fix_coral_fstring_syntax.py

Hard-refuses if the anchor isn't found exactly once (e.g. if you already
hand-edited this block).
"""
import sys

PATH = 'src/models/train_baseline.py'

old = '''    pre_d  = coral_monitor.history[0][1]
    post_d = coral_monitor.history[-1][1]
    print(f"\\n[CORAL] Covariance-distance readout, pre-training -> "
          f"post-training: {pre_d:.6f} -> {post_d:.6f}  "
          f"({'reduced' if post_d < pre_d else 'NOT reduced -- investigate '
             'before trusting this run'})")'''

new = '''    pre_d  = coral_monitor.history[0][1]
    post_d = coral_monitor.history[-1][1]
    _coral_verdict = ('reduced' if post_d < pre_d
                       else 'NOT reduced -- investigate before trusting this run')
    print(f"\\n[CORAL] Covariance-distance readout, pre-training -> "
          f"post-training: {pre_d:.6f} -> {post_d:.6f}  ({_coral_verdict})")'''

with open(PATH, 'r') as f:
    content = f.read()

n = content.count(old)
if n == 0:
    sys.exit(f"REFUSING: broken anchor not found in {PATH} -- either already "
              "fixed, or the file doesn't match what this hotfix expects. "
              "No changes written.")
if n > 1:
    sys.exit(f"REFUSING: anchor matches {n} times (expected 1). No changes written.")

content = content.replace(old, new)
with open(PATH, 'w') as f:
    f.write(content)

print(f"Patched: {PATH}")
print("Re-run the sanity check: python3 -m py_compile src/models/train_baseline.py")
