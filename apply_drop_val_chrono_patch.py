"""
apply_drop_val_chrono_patch.py
=================================
Removes X_val_chrono/y_val_chrono/domain_val_chrono from
build_dataset_multi.py -- the same fix already applied to
build_dataset_multi_longctx.py for the identical array (see that script's
own docstring), never carried over to the plain (non-longctx) script.

Root cause, confirmed via live memory monitoring (20 July 2026 session):
each pool patient's FULL chronological val slice (15% of their entire
recording -- tens of thousands of windows, zero seizures by construction
for every pool patient, since pool patients are chosen specifically
because all their seizures fall in the training region) was being copied
to float32, scaled, and appended to a growing list across ALL pool
patients, then concatenated into X_val_chrono. For a 7-patient pool this
summed to ~185,770 windows (~6.85GB as float32 alone, more during the
transient list+concatenate double-hold) -- confirmed the dominant memory
cost, not the small undersampled train/SMOTE portions. RAM was observed
pinned near the 11GB WSL2 ceiling from early in the per-patient loop, not
spiking only at the final SMOTE call.

X_val_chrono/y_val_chrono/domain_val_chrono are confirmed UNUSED
downstream -- train_baseline.py never reads them from the npz (same
confirmation already documented for the longctx sibling's identical
array). They exist only as a diagnostic/reference value that nothing
consumes. The ACTUALLY-USED X_val/y_val/domain_val (built from the small,
already-undersampled real pool, stratified-split before SMOTE) is
completely untouched by this patch.

Three changes, all in build_dataset_multi.py:
  1. Drop the per-patient X_vl_s construction (the expensive line) and its
     three list-append calls, inside the per-patient loop.
  2. Drop the three concatenate calls building X_val_chrono/y_val_chrono/
     domain_val_chrono.
  3. Drop the two save-block fields and their print line.

Run from ~/dissertation/ with akida_env activated:
    python3 apply_drop_val_chrono_patch.py

Hard-refuses (exits nonzero, writes nothing) if any anchor isn't found
exactly once.
"""
import sys

PATH = 'src/preprocessing/build_dataset_multi.py'


def patch_file(path, replacements):
    with open(path, 'r') as f:
        content = f.read()
    for i, (old, new) in enumerate(replacements):
        n = content.count(old)
        if n == 0:
            sys.exit(f"REFUSING: anchor #{i} not found in {path}.\n"
                      "File on disk doesn't match what this patch expects "
                      "-- no changes written to this file.")
        if n > 1:
            sys.exit(f"REFUSING: anchor #{i} matches {n} times in {path} "
                      "(expected exactly 1). No changes written.")
        content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Patched: {path}")


# ── 1. Per-patient loop: drop X_vl_s construction + the val_chrono appends ──
old_1 = """    # Val slice: also small enough to copy safely
    X_vl_s = np.array(X[n_train:n_train + n_val], dtype='float32')
    X_vl_s = (X_vl_s * scale + shift).clip(0, 255).astype('float32')

    train_X_parts.append(X_sub_s)
    train_y_parts.append(y_sub)
    val_X_parts.append(X_vl_s)
    val_y_parts.append(y_vl)
    domain_train_parts.append(np.full(len(y_sub), domain_id, dtype=np.int32))
    domain_val_parts.append(np.full(len(y_vl), domain_id, dtype=np.int32))

    del X, y, X_sub, X_sub_s, X_vl_s   # release mmap + copies"""

new_1 = """    # X_val_chrono construction removed here (memory fix, 20 July 2026
    # pool7/8 OOM session) -- see apply_drop_val_chrono_patch.py docstring.
    # This was the dominant memory cost: each patient's FULL val slice
    # (tens of thousands of windows) was copied to float32 and retained
    # for the whole loop, for a value confirmed unused downstream.
    train_X_parts.append(X_sub_s)
    train_y_parts.append(y_sub)
    domain_train_parts.append(np.full(len(y_sub), domain_id, dtype=np.int32))

    del X, y, X_sub, X_sub_s   # release mmap + copies"""

# ── 2. Pool section: drop the three val_chrono concatenate calls ───────────
old_2 = """X_pool       = np.concatenate(train_X_parts, axis=0)
y_pool       = np.concatenate(train_y_parts, axis=0)
X_val_chrono = np.concatenate(val_X_parts,   axis=0)   # real, chronological,
y_val_chrono = np.concatenate(val_y_parts,   axis=0)   # but 0 seizures — kept for reference only
domain_pool       = np.concatenate(domain_train_parts, axis=0)   # DANN scoping
domain_val_chrono = np.concatenate(domain_val_parts,   axis=0)"""

new_2 = """X_pool       = np.concatenate(train_X_parts, axis=0)
y_pool       = np.concatenate(train_y_parts, axis=0)
domain_pool       = np.concatenate(domain_train_parts, axis=0)   # DANN scoping
# X_val_chrono/y_val_chrono/domain_val_chrono removed here (memory fix,
# 20 July 2026 pool7/8 OOM session) -- see apply_drop_val_chrono_patch.py
# docstring for the full diagnosis. Confirmed unused downstream (same
# audit already applied to build_dataset_multi_longctx.py's identical
# array)."""

# ── 3. Save block: drop the two val_chrono fields + their print line ───────
old_3 = """np.savez_compressed(out_path,
                    X_train=X_train, y_train=y_train, domain_train=domain_train,
                    X_val=X_val,     y_val=y_val,     domain_val=domain_val,
                    X_val_chrono=X_val_chrono, y_val_chrono=y_val_chrono,
                    domain_val_chrono=domain_val_chrono)
print(f"\\nSaved  : {out_path}")
print(f"X_train: {X_train.shape}  range=[{X_train.min():.1f}, {X_train.max():.1f}]")
print(f"y_train: seizure={int(y_train.sum())} ({100*y_train.mean():.1f}%)")
print(f"X_val  : {X_val.shape}  seizure={int(y_val.sum())} ({100*y_val.mean():.1f}%)  (real, pre-SMOTE)")
print(f"X_val_chrono: {X_val_chrono.shape}  seizure={int(y_val_chrono.sum())}  (reference only)")"""

new_3 = """np.savez_compressed(out_path,
                    X_train=X_train, y_train=y_train, domain_train=domain_train,
                    X_val=X_val,     y_val=y_val,     domain_val=domain_val)
print(f"\\nSaved  : {out_path}")
print(f"X_train: {X_train.shape}  range=[{X_train.min():.1f}, {X_train.max():.1f}]")
print(f"y_train: seizure={int(y_train.sum())} ({100*y_train.mean():.1f}%)")
print(f"X_val  : {X_val.shape}  seizure={int(y_val.sum())} ({100*y_val.mean():.1f}%)  (real, pre-SMOTE)")"""


if __name__ == '__main__':
    patch_file(PATH, [
        (old_1, new_1),
        (old_2, new_2),
        (old_3, new_3),
    ])
    print("\nAll patches applied successfully.")
    print("\nSanity check:")
    print("  python3 -m py_compile src/preprocessing/build_dataset_multi.py")
