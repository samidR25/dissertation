"""
src/models/train_baseline.py
=============================
Full ANN training on local RTX 3060 via WSL2 CUDA 12.4.

Phase 2a usage (unchanged):
    # v2 primary training
    python3 src/models/train_baseline.py --patient chb01 --model-version 2

    # v2 patient-specific fine-tuning (head-only, frozen conv extractor)
    python3 src/models/train_baseline.py --patient chb03 --model-version 2 \
        --finetune-from chb01

Phase 2c additions:
    # Multi-patient base model (pools chb01+chb02+chb05)
    python3 src/models/train_baseline.py --model-version 2 --multi-patient

    # Gradual unfreezing adaptation of multi-patient base to chb03
    python3 src/models/train_baseline.py --patient chb03 --model-version 2 \
        --finetune-from multi --gradual-unfreeze

Flags:
    --multi-patient     Load from multi_dataset_ann.npz; save best_ann_multi_v2.h5
    --gradual-unfreeze  3-phase fine-tuning instead of head-only (requires
                        --finetune-from to be set)
    --model-version     1 or 2  (default 2)
    --patient           chbXX   (ignored when --multi-patient is set)
    --finetune-from     chbXX or 'multi' — base model to fine-tune from
    --class-weight      class weight multiplier for seizure class (default 1.5)
    --epochs            max epochs for initial training (not gradual unfreeze)
    --batch             batch size (default 32, safe for 6GB VRAM)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import argparse, json, os, sys
import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import confusion_matrix
from cnn2snn import check_model_compatibility, set_akida_version, AkidaVersion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions — defined first so module-level code below can call them
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_and_fit(model, X_tr, y_tr, X_vl, y_vl,
                     epochs, lr, batch, class_weight, ckpt_path):
    """Compile and fit with early stopping + LR reduction."""
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_loss',
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-7, verbose=1),
    ]
    model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=epochs,
        batch_size=batch,
        class_weight={0: 1.0, 1: class_weight},
        callbacks=callbacks,
        verbose=2,
    )


def _metrics_from_cm(y_true, y_pred):
    """Return sensitivity, specificity, FPR/hr from binary predictions."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None, None, None
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return None, None, None
    tn, fp, fn, tp = cm.ravel()
    sens   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    n_neg  = tn + fp
    fpr_hr = fp / (n_neg * 2 / 3600) if n_neg > 0 else 0.0
    return float(sens), float(spec), float(fpr_hr)

def _evaluate_and_save(model, X_tr, y_tr, X_vl, y_vl,
                        X_te, y_te, patient_tag, model_version):
    """Run evaluation on all available splits and save JSON results."""
    import tensorflow as tf
    results = {'patient': patient_tag, 'model_version': model_version}
    for split_name, X_s, y_s in [
        ('train', X_tr, y_tr),
        ('val',   X_vl, y_vl),
        ('test',  X_te, y_te),
    ]:
        if X_s is None:
            continue
        chunk, preds = 16, np.empty(len(X_s), dtype=np.int32)
        for s in range(0, len(X_s), chunk):
            preds[s:s+chunk] = np.argmax(
                model.predict(X_s[s:s+chunk], verbose=0), axis=1)
        sens, spec, fpr_hr = _metrics_from_cm(y_s, preds)
        if sens is not None:
            print(f"\n  {split_name:5s}: sens={sens:.4f}  spec={spec:.4f}  "
                  f"FPR/hr={fpr_hr:.2f}")
        else:
            print(f"\n  {split_name:5s}: insufficient classes for metrics")
        results[split_name] = {
            'n': int(len(y_s)),
            'n_seizure': int(y_s.sum()),
            'sensitivity': round(sens, 4) if sens is not None else None,
            'specificity': round(spec, 4) if spec is not None else None,
            'fpr_per_hour': round(fpr_hr, 2) if fpr_hr is not None else None,
        }
    out_path = f'results/ann_results_{patient_tag}_v{model_version}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Model: results/best_ann_{patient_tag}_v{model_version}.h5")

def _stratified_real_split(data, seed=42, test_size=0.20,
                            already_multichannel=False, data_frac=1.0):
    """
    Gate 0a pattern (originally built for --gradual-unfreeze only),
    generalised in Gate 2d to cover fresh training and --freeze-depth too.

    Splits the REAL, undersampled, pre-SMOTE train pool (X_train_real/
    y_train_real) stratified, so val is GUARANTEED to contain seizure
    windows regardless of how they fall chronologically. SMOTE applied
    only to the resulting train portion. Val is real, untouched — used
    ONLY for early-stopping/checkpoint selection, never reported as a
    held-out metric (that's still the chronological X_test).

    chb03's chronological val has ZERO seizures (all 3 fall in train/test
    by accident) — any path relying on the raw chronological val for
    ModelCheckpoint(monitor='val_loss') silently selects a checkpoint
    based on nothing but false-positive noise. chb10 never exposed this
    because its chronological val happened to contain 66 seizures.

    already_multichannel: True for --stft/--longctx data, which is already
    (N, 18, window_samples, 3) — applying [..., np.newaxis] to that would
    produce an incorrect 5D array. False (default) for the 1-channel
    baseline, which needs the newaxis to become (N, 18, window_samples, 1).

    data_frac: C2 compounding item 3 (data-efficiency sweep). When < 1.0,
    stratified-subsamples X_train_real/y_train_real to this fraction BEFORE
    the val split and BEFORE SMOTE, so the fraction is about real, labelled
    windows actually available for fine-tuning, not post-SMOTE-inflated
    counts. Default 1.0 = unchanged behaviour (no subsampling).
    """
    if 'X_train_real' not in data.files:
        sys.exit(
            "ERROR: dataset predates the Gate 0a fix (missing "
            "X_train_real/y_train_real). Regenerate via:\n"
            "  python3 src/preprocessing/build_dataset.py --patient <pid>"
        )
    if already_multichannel:
        X_train_real = data['X_train_real'].astype('float32')
    else:
        X_train_real = data['X_train_real'][..., np.newaxis].astype('float32')
    y_train_real = data['y_train_real']

    if data_frac < 1.0:
        n_seiz_full = int(y_train_real.sum())
        if n_seiz_full < 2:
            sys.exit(f"ERROR: only {n_seiz_full} seizure window(s) in the "
                     "real train pool BEFORE subsampling — cannot "
                     "stratified-subsample.")
        X_train_real, _, y_train_real, _ = train_test_split(
            X_train_real, y_train_real, train_size=data_frac,
            stratify=y_train_real, random_state=seed,
        )
        print(f"\n  [Item 3: data-efficiency sweep] Subsampled real train "
              f"pool to {data_frac:.0%} ({len(y_train_real)} windows, "
              f"{int(y_train_real.sum())} seizure) BEFORE val split/SMOTE.")

    n_seiz = int(y_train_real.sum())
    if n_seiz < 2:
        sys.exit(f"ERROR: only {n_seiz} seizure window(s) in the real "
                 "train pool — cannot stratified-split for a seizure-"
                 "guaranteed val.")

    X_tr_real, X_vl, y_tr_real, y_vl = train_test_split(
        X_train_real, y_train_real, test_size=test_size,
        stratify=y_train_real, random_state=seed,
    )
    n_seiz_tr = int(y_tr_real.sum())
    if n_seiz_tr < 2:
        sys.exit(f"ERROR: only {n_seiz_tr} seizure window(s) left in "
                 "train after the split — cannot SMOTE.")

    flat_shape = X_tr_real.shape
    X_flat = X_tr_real.reshape(len(X_tr_real), -1)
    sm = SMOTE(random_state=seed, k_neighbors=min(5, n_seiz_tr - 1))
    X_res, y_res = sm.fit_resample(X_flat, y_tr_real)
    X_tr = X_res.reshape((-1,) + flat_shape[1:]).astype('float32')
    y_tr = y_res.astype('int32')

    print(f"\n  Seizure-guaranteed stratified split (Gate 2d):")
    print(f"  Train (post-SMOTE)     : {len(X_tr):>6}  seizure={int(y_tr.sum())}")
    print(f"  Val   (real, untouched): {len(X_vl):>6}  seizure={int(y_vl.sum())}")
    return X_tr, y_tr, X_vl, y_vl

def _write_ckpt_manifest(ckpt_path, patient_tag, seed, finetune_from,
                          gradual_unfreeze, model_version, freeze_depth=None):
    """Gate 1b: sidecar manifest — which scaler this checkpoint's training
    data used, so eval-time loaders can refuse on mismatch (§5 regression
    test). Local import: src.manifest needs '.' on sys.path, which happens
    later in this script (Section 2) — safe here since this only runs at
    the very end of training."""
    import sys as _sys
    _sys.path.insert(0, '.')
    from src.manifest import write_manifest
    if '_longctx_w' in patient_tag:
        _base_tag, _ws = patient_tag.split('_longctx_w')
        scaler_path = (f'data/processed/multi_scaler_longctx_w{_ws}.json'
                       if _base_tag == 'multi' or _base_tag.startswith('multi_from_')
                       or _base_tag.startswith('multi_sslpretrain')
                       else f'data/processed/{_base_tag}_scaler_longctx_w{_ws}.json')
    elif patient_tag.endswith('_g'):
        # Candidate G (sec8) -- without this branch, patient_tag='multi_g'
        # would fall through to the generic else-branch below and resolve
        # to data/processed/multi_g_scaler.json, which does not exist --
        # build_dataset_g.py writes multi_scaler_g.json / {tag}_scaler_g.json.
        _base_tag = patient_tag[:-len('_g')]
        scaler_path = ('data/processed/multi_scaler_g.json'
                       if _base_tag == 'multi' or _base_tag.startswith('multi_from_')
                       or _base_tag.startswith('multi_sslpretrain')
                       else f'data/processed/{_base_tag}_scaler_g.json')
    else:
        scaler_path = ('data/processed/multi_scaler.json'
                       if patient_tag == 'multi' or patient_tag.startswith('multi_from_')
                       or patient_tag.startswith('multi_sslpretrain')
                       else f'data/processed/{patient_tag}_scaler.json')
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path) as f:
            scaler = json.load(f)
    else:
        print(f"WARNING: {scaler_path} not found — checkpoint manifest will "
              "have no scaler provenance.")

    write_manifest(
        ckpt_path,
        patient_tag=patient_tag,
        seed=seed,
        scaler_path=scaler_path,
        scaler=scaler,
        finetune_from=finetune_from,
        gradual_unfreeze=bool(gradual_unfreeze),
        freeze_depth=freeze_depth,
        model_version=model_version,
        )
    print(f"Manifest saved: {ckpt_path}.manifest.json")

def _run_gradual_unfreeze(base_model, X_tr, y_tr, X_vl, y_vl,
                           patient_tag, model_version, class_weight, batch,
                           X_test=None, y_test=None):
    """
    3-phase gradual unfreezing for chb03 cross-patient adaptation.

    Layer naming convention (v2 architecture):
      rescaling, conv1, pool1, relu1, conv2, pool2, relu2,
      conv3, pool3, relu3, flatten, dense1, relu_dense, output
    conv1 = (9,7) spatio-temporal — stays frozen throughout.
    """
    ckpt_base = f'results/best_ann_{patient_tag}_v{model_version}.h5'
# ── Phase 1: Dense head only (conv extractor fully frozen) ───────────────
    print("\n--- Gradual Unfreeze Phase 1: Dense head only ---")
    print("  Freezing: conv1, conv2, conv3 | Trainable: flatten + dense layers")
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers:
        if any(layer.name.startswith(p) for p in ('dense', 'flatten', 'output', 'relu_dense')):
            layer.trainable = True
        if layer.name.startswith('bn'):          # Gate 2b fix
            layer.trainable = True
    n_trainable = sum(p.numpy().size for p in base_model.trainable_variables)
    print(f"  Trainable params: {n_trainable:,}")
    _compile_and_fit(base_model, X_tr, y_tr, X_vl, y_vl,
                     epochs=20, lr=1e-4, batch=batch,
                     class_weight=class_weight, ckpt_path=ckpt_base)
    best_model = keras.models.load_model(ckpt_base)

    # ── Phase 2: Unfreeze conv3 ───────────────────────────────────────────────
    print("\n--- Gradual Unfreeze Phase 2: conv3 + head ---")
    print("  Freezing: conv1, conv2 | Trainable: conv3 + flatten + dense layers")
    for layer in best_model.layers:
        layer.trainable = False
    for layer in best_model.layers:
        if any(layer.name.startswith(p)
               for p in ('conv3', 'pool3', 'relu3', 'dense', 'flatten', 'output', 'relu_dense')):
            layer.trainable = True
    n_trainable = sum(p.numpy().size for p in best_model.trainable_variables)
    print(f"  Trainable params: {n_trainable:,}")
    _compile_and_fit(best_model, X_tr, y_tr, X_vl, y_vl,
                     epochs=20, lr=5e-5, batch=batch,
                     class_weight=class_weight, ckpt_path=ckpt_base)
    best_model = keras.models.load_model(ckpt_base)

    # ── Phase 3: Unfreeze conv2 + conv3 (conv1 stays frozen) ─────────────────
    print("\n--- Gradual Unfreeze Phase 3: conv2 + conv3 + head ---")
    print("  Freezing: conv1 (9,7 spatio-temporal — stays frozen throughout)")
    print("  Trainable: conv2, conv3, flatten + dense layers")
    for layer in best_model.layers:
        layer.trainable = False
    for layer in best_model.layers:
        if any(layer.name.startswith(p)
               for p in ('conv2', 'pool2', 'relu2',
                         'conv3', 'pool3', 'relu3',
                         'dense', 'flatten', 'output', 'relu_dense')):
            layer.trainable = True
    n_trainable = sum(p.numpy().size for p in best_model.trainable_variables)
    print(f"  Trainable params: {n_trainable:,}")
    _compile_and_fit(best_model, X_tr, y_tr, X_vl, y_vl,
                     epochs=30, lr=1e-5, batch=batch,
                     class_weight=class_weight, ckpt_path=ckpt_base)
    final_model = keras.models.load_model(ckpt_base)

    print(f"\nGradual unfreeze complete.  Best checkpoint: {ckpt_base}")
    _evaluate_and_save(final_model, X_tr, y_tr, X_vl, y_vl,
                       X_test, y_test,
                       patient_tag=patient_tag, model_version=model_version)

def _apply_freeze_depth(model, freeze_depth):
    """
    Gate 2a/2b: freeze the first `freeze_depth` conv blocks' KERNELS, leave
    everything else (later conv blocks + dense head) trainable. BatchNorm
    layers are ALWAYS trainable=True, in every block, regardless of
    freeze_depth — a frozen BN layer would otherwise retain the POOL's
    running mean/variance, the exact distribution being adapted away from.
    Freezing a conv kernel does not mean freezing the BN statistics
    computed on top of it.
    """
    assert 0 <= freeze_depth <= 3, f"freeze_depth must be 0..3, got {freeze_depth}"

    conv_blocks = {1: 'conv1', 2: 'conv2', 3: 'conv3'}
    frozen_conv_prefixes = tuple(conv_blocks[d] for d in range(1, freeze_depth + 1))

    print(f"\n[Gate 2a/2b] freeze_depth={freeze_depth}")
    for layer in model.layers:
        if layer.name.startswith('bn'):
            layer.trainable = True   # always — see docstring
        elif any(layer.name.startswith(p) for p in frozen_conv_prefixes):
            layer.trainable = False
        else:
            layer.trainable = True

    frozen_names = [l.name for l in model.layers if not l.trainable]
    bn_names = [l.name for l in model.layers if l.name.startswith('bn')]
    print(f"  Frozen (conv kernels only): {frozen_names if frozen_names else '(none)'}")
    print(f"  BN layers re-estimating target stats: {bn_names}")
    n_trainable = sum(p.numpy().size for p in model.trainable_variables)
    print(f"  Trainable params: {n_trainable:,}")


def _run_freeze_depth_finetune(base_model, X_tr, y_tr, X_vl, y_vl,
                                patient_tag, model_version, freeze_depth,
                                class_weight, batch, epochs, lr,
                                X_test=None, y_test=None):
    """
    Gate 2: single-phase fine-tune with a configurable freeze depth.
    Replaces the old fixed 3-phase schedule for personalisation. BN
    layers are always re-trainable (Gate 2b) — captured here so the
    pass condition (BN stats measurably moved) is verifiable directly
    from this function's own output.
    """
    ckpt_path = f'results/best_ann_{patient_tag}_v{model_version}.h5'

    # Capture base model's BN running means BEFORE fine-tuning — Gate 2b's
    # pass condition needs a before/after comparison.
    base_bn_means = {
        l.name: l.get_weights()[2].copy()   # [gamma, beta, moving_mean, moving_var]
        for l in base_model.layers if l.name.startswith('bn')
    }

    _apply_freeze_depth(base_model, freeze_depth)

    _compile_and_fit(
        base_model, X_tr, y_tr, X_vl, y_vl,
        epochs=epochs, lr=lr, batch=batch,
        class_weight=class_weight, ckpt_path=ckpt_path,
    )

    best_model = keras.models.load_model(ckpt_path)

    print(f"\n[Gate 2b] BN running-mean shift vs. base model:")
    for l in best_model.layers:
        if l.name.startswith('bn') and l.name in base_bn_means:
            new_mean = l.get_weights()[2]
            delta = np.abs(new_mean - base_bn_means[l.name]).mean()
            print(f"  {l.name}: mean |Δrunning_mean| = {delta:.6f}")

    if X_test is not None:
        _evaluate_and_save(best_model, X_tr, y_tr, X_vl, y_vl,
                            X_test, y_test, patient_tag, model_version)

    return best_model

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--patient',          default='chb01')
parser.add_argument('--model-version',    type=int, default=2)
parser.add_argument('--finetune-from',    default=None,
                    help="Patient tag or 'multi' — loads best_ann_<tag>_v2.h5 as base")
parser.add_argument('--multi-patient',    action='store_true',
                    help="Train on pooled multi-patient dataset (multi_dataset_ann.npz)")
parser.add_argument('--gradual-unfreeze', action='store_true',
                    help="3-phase gradual unfreezing (requires --finetune-from)")
parser.add_argument('--freeze-depth', type=int, default=None,
                    help="Freeze the first N conv blocks' KERNELS during "
                         "fine-tuning (0=full fine-tune .. 3=head-only). "
                         "BatchNorm layers are always trainable regardless "
                         "of this value (Gate 2b). Mutually exclusive with "
                         "--gradual-unfreeze — if both are omitted, falls "
                         "back to head-only fine-tuning (Phase 2a behaviour).")
parser.add_argument('--finetune-data-frac', type=float, default=1.0,
                    help="C2 compounding item 3 (data-efficiency sweep): "
                         "stratified-subsample the patient's real, "
                         "pre-SMOTE fine-tuning pool (X_train_real/"
                         "y_train_real) to this fraction (e.g. 0.25) "
                         "BEFORE the val split and BEFORE SMOTE. Default "
                         "1.0 = unchanged behaviour. Requires "
                         "--finetune-from; refused otherwise (fresh/pool "
                         "training must never be silently subsampled).")
parser.add_argument('--class-weight',     type=float, default=1.5)
parser.add_argument('--epochs',           type=int, default=100)
parser.add_argument('--batch',            type=int, default=32)
parser.add_argument('--stft',             action='store_true',
                    help='Use 3-channel STFT dataset (Phase 2d)')
parser.add_argument('--longctx',          action='store_true',
                    help='Use 3-channel long-context dataset (Gate 2 — '
                         'rolling line-length + rolling delta/beta ratio). '
                         'Mutually exclusive with --stft.')
parser.add_argument('--g-features',       action='store_true',
                    help='Candidate G phase 1 (Handoff_a_e_c_closed_to_'
                         'next_steps.md sec8): 3-channel relative-band-'
                         'power dataset (relative delta power, relative '
                         'beta power, delta/beta ratio) -- REPLACES raw '
                         'EEG entirely, does not augment it. Requires a '
                         'dataset built via build_dataset_g.py and '
                         '--multi-patient (compared against the frozen C1 '
                         'checkpoint on the same six-patient set, same as '
                         '--init-from-ssl). Mutually exclusive with '
                         '--stft/--longctx/--dann/--coral/--init-from-ssl/'
                         '--finetune-from (first screen only).')
parser.add_argument('--window-samples',   type=int, default=512,
                    choices=[512, 768],
                    help='Window size in samples. Only meaningful with '
                         '--longctx (Arm B=512, Arm C=768); --stft and the '
                         '1-channel baseline path are always 512.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--dann', action='store_true',
                    help='DANN scoping experiment (Handoff_calibration_'
                         'session_to_dann_supcon.md sec4d/4e). Requires '
                         '--multi-patient and a dataset built with domain '
                         'labels (domain_train/domain_val in the npz). '
                         'Mutually exclusive with --finetune-from, --stft, '
                         '--longctx.')
parser.add_argument('--dann-lambda', type=float, default=0.1,
                    help='Domain-loss weight / GRL lambda for --dann '
                         '(default 0.1; first screen also tries 0.5)')
parser.add_argument('--coral', action='store_true',
                    help='CORAL (correlation alignment) scoping experiment, '
                         'A-i multi-source variant (Handoff_post_dann_'
                         'scoping_to_implementation.md sec3). Requires '
                         '--multi-patient and a dataset built with domain '
                         'labels. Mutually exclusive with --dann (run '
                         'separately for direct comparability), '
                         '--finetune-from, --stft, --longctx.')
parser.add_argument('--coral-lambda', type=float, default=0.01,
                    help='CORAL loss weight (default 0.01; first screen '
                         'also tries 0.1 -- CORAL losses sit on a very '
                         'different scale than cross-entropy, do not reuse '
                         'DANN lambda values, sec3d)')
parser.add_argument('--init-from-ssl', default=None,
                    help='Candidate C (Handoff_post_dann_scoping_to_'
                         'implementation.md sec5d): path to a trunk '
                         'checkpoint produced by pretrain_ssl.py (Dense '
                         'head still at random init). Loaded INSTEAD OF '
                         'fresh random init, then normal multi-patient '
                         'supervised training proceeds unchanged. Requires '
                         '--multi-patient. Mutually exclusive with '
                         '--finetune-from/--dann/--coral/--stft/--longctx '
                         '(first screen only).')
args = parser.parse_args()

if args.dann and not args.multi_patient:
    parser.error('--dann requires --multi-patient (domains = pool patients)')
if args.dann and (args.finetune_from or args.stft or args.longctx):
    parser.error('--dann is mutually exclusive with --finetune-from/--stft/--longctx '
                 '(first screen only)')
if args.coral and not args.multi_patient:
    parser.error('--coral requires --multi-patient (domains = pool patients)')
if args.coral and (args.finetune_from or args.stft or args.longctx):
    parser.error('--coral is mutually exclusive with --finetune-from/--stft/--longctx '
                 '(first screen only)')
if args.dann and args.coral:
    parser.error('--dann and --coral are mutually exclusive -- run separately '
                 'for direct, uncomplicated comparability against the closed '
                 'DANN table')
if args.init_from_ssl and not args.multi_patient:
    parser.error('--init-from-ssl requires --multi-patient (sec5e: compared '
                 'against the frozen C1 checkpoint on the same six-patient set)')
if args.init_from_ssl and (args.finetune_from or args.dann or args.coral
                            or args.stft or args.longctx):
    parser.error('--init-from-ssl is mutually exclusive with --finetune-from/'
                 '--dann/--coral/--stft/--longctx (first screen only)')
if args.g_features and not args.multi_patient:
    parser.error('--g-features requires --multi-patient (sec8 phase 1: '
                 'compared against the frozen C1 checkpoint on the same '
                 'six-patient set, same requirement as --init-from-ssl)')
if args.g_features and (args.finetune_from or args.dann or args.coral
                         or args.stft or args.longctx or args.init_from_ssl):
    parser.error('--g-features is mutually exclusive with --finetune-from/'
                 '--dann/--coral/--stft/--longctx/--init-from-ssl (first '
                 'screen only)')
# ── Deterministic seed ────────────────────────────────────────────────────────
import random, tensorflow as tf
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
print(f"Random seed: {args.seed}")

# Validation
if args.gradual_unfreeze and args.finetune_from is None:
    parser.error("--gradual-unfreeze requires --finetune-from")
if args.finetune_data_frac < 1.0 and args.finetune_from is None:
    parser.error("--finetune-data-frac requires --finetune-from -- fresh/"
                 "pool training must never be silently subsampled by this "
                 "flag.")
if not (0.0 < args.finetune_data_frac <= 1.0):
    parser.error("--finetune-data-frac must be in (0.0, 1.0]")
if args.multi_patient and args.gradual_unfreeze:
    parser.error("--multi-patient trains a base model; "
                 "use --gradual-unfreeze only when fine-tuning (--finetune-from multi)")
if args.stft and args.longctx:
    parser.error("--stft and --longctx are mutually exclusive")
if args.window_samples != 512 and not args.longctx:
    parser.error("--window-samples is only meaningful with --longctx")

# Derive tag used in file names
# new:
if args.multi_patient and args.finetune_from:
    patient_tag = f'multi_from_{args.finetune_from}'
elif args.multi_patient:
    patient_tag = 'multi'
else:
    patient_tag = args.patient
if args.stft:
    patient_tag += '_stft'
if args.longctx:
    patient_tag += f'_longctx_w{args.window_samples}'
if args.g_features:
    patient_tag += '_g'
if args.init_from_ssl:
    patient_tag += '_sslpretrain'
# ── GPU memory growth ──────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
print(f"Training on: {gpus[0].name if gpus else 'CPU (no GPU found)'}")

# ── 1. Load data ───────────────────────────────────────────────────────────────
if args.multi_patient:
    data_path = 'data/processed/multi_dataset_ann.npz'
else:
    data_path = f'data/processed/{args.patient}_dataset_ann.npz'
# NOTE: existence of the 1-channel default data_path above is intentionally
# NOT checked here — --stft/--longctx below override data_path entirely
# before it's ever used, and checking the (irrelevant) 1-channel default's
# existence first would wrongly block --stft/--longctx runs on machines
# that never built the 1-channel dataset for this patient/pool. The real
# check happens once, after all overrides, right before np.load(data_path).

if args.stft:
    stft_path = data_path.replace('_dataset_ann.npz', '_dataset_stft.npz')
    if not os.path.exists(stft_path):
        sys.exit(f"ERROR: {stft_path} not found.\n"
                 f"Run: python3 src/preprocessing/build_dataset_stft.py --patient {args.patient}")
    data_path = stft_path

if args.longctx:
    longctx_path = data_path.replace(
        '_dataset_ann.npz', f'_dataset_longctx_w{args.window_samples}.npz')
    if not os.path.exists(longctx_path):
        sys.exit(
            f"ERROR: {longctx_path} not found.\n"
            f"Run first:\n"
            f"  python3 src/preprocessing/preprocess.py --patient {args.patient} "
            f"--longctx --window-s <2.0 or 3.0> --longctx-lookback-s 12\n"
            f"  python3 src/preprocessing/build_dataset_longctx.py "
            f"--patient {args.patient} --window-samples {args.window_samples}"
        )
    data_path = longctx_path

if args.g_features:
    g_path = data_path.replace('_dataset_ann.npz', '_dataset_g.npz')
    if not os.path.exists(g_path):
        sys.exit(f"ERROR: {g_path} not found.\n"
                 f"Run: python3 src/preprocessing/build_dataset_g.py --multi-patient")
    data_path = g_path

if not os.path.exists(data_path):
    sys.exit(f"ERROR: {data_path} not found — build it first for this "
             "configuration (1ch: build_dataset.py/build_dataset_multi.py; "
             "stft: build_dataset_stft.py; longctx: preprocess.py --longctx "
             "+ build_dataset_longctx.py/build_dataset_multi_longctx.py).")

data    = np.load(data_path)
if args.stft or args.longctx or args.g_features:
    X_train = data['X_train'].astype('float32')   # already (N, 18, ws, 3)
    X_val   = data['X_val'].astype('float32')
else:
    X_train = data['X_train'][..., np.newaxis].astype('float32')
    X_val   = data['X_val']  [..., np.newaxis].astype('float32')
y_train = data['y_train']
y_val   = data['y_val']

# Test split: present for per-patient datasets, absent for multi
has_test = 'X_test' in data.files
if has_test:
    X_test = data['X_test'].astype('float32') if (args.stft or args.longctx or args.g_features) \
             else data['X_test'][..., np.newaxis].astype('float32')
    y_test = data['y_test']

# ── DANN / CORAL domain labels ────────────────────────────────────────────────
if args.dann or args.coral:
    if 'domain_train' not in data.files or 'domain_val' not in data.files:
        sys.exit(
            f"ERROR: {data_path} has no domain_train/domain_val arrays -- "
            "it predates the DANN-scoping patch to build_dataset_multi.py.\n"
            "Rebuild via:\n"
            "  python3 src/preprocessing/build_dataset_multi.py "
            "--patients chb01 chb02 chb05"
        )
    domain_train = data['domain_train']
    domain_val   = data['domain_val']
    n_domains    = int(max(domain_train.max(), domain_val.max()) + 1)
    print(f"\n[domain-labels] domain_train: {len(domain_train)}  "
          f"domain_val: {len(domain_val)}  n_domains={n_domains}")

# ── Multi-patient val fix ──────────────────────────────────────────────────────
# The npz val split (chron 70–85%) contains zero seizure windows for all three
# patients — all ictal recordings fall in the first 70% chronologically.
# A val set with no seizures gives early stopping and checkpointing a useless
# signal: it optimises for non-seizure accuracy and may stop before the model
# learns ictal features properly.
#
# Fix: carve 20% of the post-SMOTE training set as a stratified val set.
# Result: balanced val (~50% seizure, ~4750 windows) that stops training when
# seizure classification starts to overfit, not when non-seizure accuracy peaks.
# The npz X_val is discarded for this case only.
#
# This does NOT affect per-patient training (--patient flag): those datasets
# have properly distributed val splits already.
if args.multi_patient:
    # X_val/y_val loaded from the npz are now a real, non-synthetic split,
    # carved BEFORE SMOTE in build_dataset_multi.py — safe to use as-is.
    print(f"\nDataset: {data_path}")
    print(f"  Train : {len(X_train):>6}  seizure={int(y_train.sum())} "
          f"({100*y_train.mean():.1f}%)")
    print(f"  Val   : {len(X_val):>6}  seizure={int(y_val.sum())} "
          f"({100*y_val.mean():.1f}%)  (real, pre-SMOTE split)")
    if has_test:
        print(f"  Test  : {len(X_test):>6}  seizure={int(y_test.sum())}  "
              f"(held-out — evaluated after training)")
elif args.gradual_unfreeze:
    # ── GATE 0a FIX (Phase 3 Session 2) ──────────────────────────────────
    # OLD (leaky) behaviour carved val via train_test_split on data['X_train'],
    # which build_dataset.py saves ALREADY SMOTE'd. A split applied after SMOTE
    # can put a synthetic point in val and its real source neighbour in train
    # (or vice versa) — the same §3 leak, just unfixed in this path until now.
    #
    # FIX (identical pattern to build_dataset_multi.py's §3 fix): use the REAL,
    # pre-SMOTE, undersampled train pool (X_train_real/y_train_real). Split
    # THAT first (stratified 80/20) — val is real and untouched. SMOTE is then
    # applied only to the resulting train portion, here, at runtime.
    if 'X_train_real' not in data.files:
        sys.exit(
            f"ERROR: {data_path} predates the Gate 0a SMOTE-before-split fix "
            "(missing X_train_real/y_train_real).\n"
            f"Regenerate it first:\n"
            f"  python3 src/preprocessing/build_dataset.py --patient {args.patient}\n"
            "then re-run this command."
        )
    X_train_real = data['X_train_real'][..., np.newaxis].astype('float32')
    y_train_real = data['y_train_real']

    from sklearn.model_selection import train_test_split
    X_tr_real, X_val, y_tr_real, y_val = train_test_split(
        X_train_real, y_train_real, test_size=0.20,
        stratify=y_train_real, random_state=args.seed,
    )

    n_seiz_tr = int(y_tr_real.sum())
    if n_seiz_tr < 2:
        sys.exit(
            f"ERROR: only {n_seiz_tr} seizure window(s) left in the real train "
            "portion after the 80/20 split — cannot SMOTE. This patient has "
            "too few train-split seizures for --gradual-unfreeze."
        )
    from imblearn.over_sampling import SMOTE
    flat_shape = X_tr_real.shape
    X_flat = X_tr_real.reshape(len(X_tr_real), -1)
    sm = SMOTE(random_state=args.seed, k_neighbors=min(5, n_seiz_tr - 1))
    X_res, y_res = sm.fit_resample(X_flat, y_tr_real)
    X_train = X_res.reshape((-1,) + flat_shape[1:]).astype('float32')
    y_train = y_res.astype('int32')

    print(f"\nDataset: {data_path}")
    print("  Gate 0a leak-free split — real undersampled pool split BEFORE SMOTE:")
    print(f"  Train (post-SMOTE)     : {len(X_train):>6}  seizure={int(y_train.sum())} "
          f"({100*y_train.mean():.1f}%)")
    print(f"  Val   (real, untouched): {len(X_val):>6}  seizure={int(y_val.sum())} "
          f"({100*y_val.mean():.1f}%)")
    if has_test:
        print(f"  Test  : {len(X_test):>6}  seizure={int(y_test.sum())}  "
              f"(held-out — evaluated after training)")

elif args.stft:
    # UNVERIFIED / OUT OF SCOPE for Gate 0a (Part C carries this forward).
    # May carry the same leak gradual_unfreeze just had fixed above.
    # Left as-is deliberately — fix only if/when the STFT path is revived.
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=42,
    )
    print(f"\nDataset: {data_path}")
    print(f"  Val carved from training set (stratified 80/20 — npz val had 0 seizures):")
    print(f"  [WARNING: --stft leak fix is OUT OF SCOPE per Part C — do not trust "
          f"these val numbers for a headline result]")
    print(f"  Train : {len(X_train):>6}  seizure={int(y_train.sum())} "
          f"({100*y_train.mean():.1f}%)")
    print(f"  Val   : {len(X_val):>6}  seizure={int(y_val.sum())} "
          f"({100*y_val.mean():.1f}%)")
    if has_test:
        print(f"  Test  : {len(X_test):>6}  seizure={int(y_test.sum())}  "
              f"(held-out — evaluated after training)")

else:
    # Gate 2d: fresh training and --freeze-depth fine-tuning both fall
    # through to here — neither was protected by the Gate 0a fix before
    # now. Same seizure-guaranteed split as gradual_unfreeze already had.
    print(f"\nDataset: {data_path}")
    # Item 3: data_frac only ever applies to fine-tuning (argparse already
    # refuses --finetune-data-frac<1.0 without --finetune-from, this is
    # defence in depth so fresh/pool training is never subsampled even if
    # that check is ever bypassed or the flag's default changes upstream).
    _data_frac = args.finetune_data_frac if args.finetune_from else 1.0
    X_train, y_train, X_val, y_val = _stratified_real_split(
        data, seed=args.seed, already_multichannel=(args.stft or args.longctx),
        data_frac=_data_frac)

if args.longctx:
    expected_shape = (18, args.window_samples, 3)
elif args.stft or args.g_features:
    expected_shape = (18, 512, 3)
else:
    expected_shape = (18, 512, 1)
assert X_train.shape[1:] == expected_shape, \
    f"Wrong input shape {X_train.shape[1:]} — expected {expected_shape}"

# ── 2. Build or load model ────────────────────────────────────────────────────
sys.path.insert(0, '.')
if args.stft or args.g_features:
    from src.models.akida_cnn_v2 import build_seizure_cnn_v2_3ch
    build_fn = lambda: build_seizure_cnn_v2_3ch(n_channels=18, window_samples=512)
elif args.longctx:
    from src.models.akida_cnn_v2 import build_seizure_cnn_v2_3ch
    build_fn = lambda: build_seizure_cnn_v2_3ch(n_channels=18, window_samples=args.window_samples)
elif args.model_version == 1:
    from src.models.akida_cnn import build_seizure_cnn
    build_fn = lambda: build_seizure_cnn(n_channels=18, window_samples=512)
elif args.model_version == 2:
    from src.models.akida_cnn_v2 import (build_seizure_cnn_v2,
                                          build_patient_adapted_model)
    build_fn = lambda: build_seizure_cnn_v2(n_channels=18, window_samples=512)
else:
    sys.exit(f"Unknown --model-version {args.model_version}")
os.makedirs('results', exist_ok=True)

if args.dann:
    # ── DANN training path (first concrete experiment, Handoff sec4d/4e) ──
    from src.models.akida_cnn_v2 import (build_seizure_cnn_v2_dann,
                                          extract_deployable_submodel,
                                          GradientReversalLayer)
    print(f"\n=== DANN SCOPING EXPERIMENT (lambda={args.dann_lambda}) ===")
    print(f"Domains: {n_domains} (pool patients, in --patients order)")

    dann_model = build_seizure_cnn_v2_dann(
        n_channels=18, window_samples=512,
        n_domains=n_domains, grl_lambda=args.dann_lambda)

    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(dann_model)
        print("[DANN] compat check ran on the branching model -- not "
              "meaningful (it will never be converted). The DEPLOYABLE "
              "submodel gets the real check after extraction, below.")
    except Exception as e:
        print(f"[DANN] compat check on branching model raised: {e} "
              "(expected/ignorable -- see extract_deployable_submodel step)")

    dann_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={'output': 'sparse_categorical_crossentropy',
              'domain_output': 'sparse_categorical_crossentropy'},
        loss_weights={'output': 1.0, 'domain_output': 1.0},
        metrics={'output': 'accuracy', 'domain_output': 'accuracy'},
    )
    dann_model.summary(print_fn=lambda x: print(f"  {x}"))

    dann_ckpt = f'results/best_ann_{patient_tag}_v{args.model_version}_dann_lambda{args.dann_lambda}_TRAINING.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            dann_ckpt, monitor='val_output_loss',
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_output_loss', patience=15,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_output_loss', factor=0.5,
            patience=7, min_lr=1e-7, verbose=1),
    ]
    dann_model.fit(
        X_train, {'output': y_train, 'domain_output': domain_train},
        validation_data=(X_val, {'output': y_val, 'domain_output': domain_val}),
        epochs=args.epochs, batch_size=args.batch,
        callbacks=callbacks, verbose=2,
    )

    best_dann = keras.models.load_model(
        dann_ckpt, custom_objects={'GradientReversalLayer': GradientReversalLayer})

    print("\n[DANN] Extracting deployable submodel (trunk + main head only)...")
    deployable = extract_deployable_submodel(best_dann, n_channels=18, window_samples=512)

    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(deployable)
        print("AKD1000 v1 compatible (deployable submodel) ✓")
    except Exception as e:
        print(f"WARNING: compatibility check on deployable submodel raised: {e}")

    # Naming matches convert_to_snn.py's --base/--variant convention exactly:
    #   results/best_ann_<base>_v<V>_<variant>.h5
    # so conversion is a plain: --base multi --variant dann_lambda<L>
    deploy_ckpt = f'results/best_ann_{patient_tag}_v{args.model_version}_dann_lambda{args.dann_lambda}.h5'
    deployable.save(deploy_ckpt)
    print(f"Saved deployable checkpoint: {deploy_ckpt}")

    _evaluate_and_save(
        deployable, X_train, y_train, X_val, y_val,
        X_test if has_test else None, y_test if has_test else None,
        patient_tag=f'{patient_tag}_dann_lambda{args.dann_lambda}',
        model_version=args.model_version,
    )
    # patient_tag passed UNMODIFIED here (not the dann-suffixed variant) so
    # _write_ckpt_manifest resolves the correct multi_scaler.json path.
    _write_ckpt_manifest(
        deploy_ckpt, patient_tag=patient_tag,
        seed=args.seed, finetune_from=None, gradual_unfreeze=False,
        model_version=args.model_version,
    )
    print(f"\n[DANN] Done. Next -- convert + eval on each held-out patient:")
    print(f"  python3 src/models/convert_to_snn.py --patient multi --base multi "
          f"--variant dann_lambda{args.dann_lambda} --eval-patient chb03 "
          f"--model-version {args.model_version} --cal-samples 1024")
    print(f"  # repeat --eval-patient for chb10 chb13 chb15 chb16 chb20")
    print(f"  python3 src/evaluation/eval_event_level.py --fbz "
          f"results/seizure_model_multi_dann_lambda{args.dann_lambda}"
          f"_v{args.model_version}_w4a4.fbz --eval-patient chb03  # repeat per patient")
    sys.exit(0)

if args.coral:
    # ── CORAL training path (A-i multi-source, sec3d first screen) ────────
    from src.models.akida_cnn_v2 import (build_seizure_cnn_v2_coral,
                                          extract_deployable_submodel,
                                          make_coral_loss,
                                          CoralDistanceMonitor)
    print(f"\n=== CORAL SCOPING EXPERIMENT (A-i multi-source, "
          f"lambda={args.coral_lambda}) ===")
    print(f"Domains: {n_domains} (pool patients, in --patients order)")

    coral_model = build_seizure_cnn_v2_coral(n_channels=18, window_samples=512)

    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(coral_model)
        print("[CORAL] compat check ran on the two-output model -- not "
              "meaningful (it will never be converted). The DEPLOYABLE "
              "submodel gets the real check after extraction, below.")
    except Exception as e:
        print(f"[CORAL] compat check on two-output model raised: {e} "
              "(expected/ignorable -- see extract_deployable_submodel step)")

    coral_loss_fn = make_coral_loss(n_domains=n_domains)
    coral_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={'output': 'sparse_categorical_crossentropy',
              'flatten': coral_loss_fn},
        loss_weights={'output': 1.0, 'flatten': args.coral_lambda},
        metrics={'output': 'accuracy'},
    )
    coral_model.summary(print_fn=lambda x: print(f"  {x}"))

    coral_monitor = CoralDistanceMonitor(coral_model, X_val, domain_val, n_domains)

    coral_ckpt = f'results/best_ann_{patient_tag}_v{args.model_version}_coral_lambda{args.coral_lambda}_TRAINING.h5'
    callbacks = [
        coral_monitor,
        keras.callbacks.ModelCheckpoint(
            coral_ckpt, monitor='val_output_loss',
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_output_loss', patience=15,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_output_loss', factor=0.5,
            patience=7, min_lr=1e-7, verbose=1),
    ]
    coral_model.fit(
        X_train, {'output': y_train, 'flatten': domain_train},
        validation_data=(X_val, {'output': y_val, 'flatten': domain_val}),
        epochs=args.epochs, batch_size=args.batch,
        callbacks=callbacks, verbose=2,
    )

    # compile=False on load: architecture+weights only, no custom-loss
    # deserialization risk -- extract_deployable_submodel() only ever
    # needs get_layer()/get_weights(), never the compiled state.
    best_coral = keras.models.load_model(coral_ckpt, compile=False)

    print("\n[CORAL] Extracting deployable submodel (trunk + main head only)...")
    deployable = extract_deployable_submodel(best_coral, n_channels=18, window_samples=512)

    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(deployable)
        print("AKD1000 v1 compatible (deployable submodel) ✓")
    except Exception as e:
        print(f"WARNING: compatibility check on deployable submodel raised: {e}")

    deploy_ckpt = f'results/best_ann_{patient_tag}_v{args.model_version}_coral_lambda{args.coral_lambda}.h5'
    deployable.save(deploy_ckpt)
    print(f"Saved deployable checkpoint: {deploy_ckpt}")

    pre_d  = coral_monitor.history[0][1]
    post_d = coral_monitor.history[-1][1]
    _coral_verdict = ('reduced' if post_d < pre_d
                       else 'NOT reduced -- investigate before trusting this run')
    print(f"\n[CORAL] Covariance-distance readout, pre-training -> "
          f"post-training: {pre_d:.6f} -> {post_d:.6f}  ({_coral_verdict})")

    _evaluate_and_save(
        deployable, X_train, y_train, X_val, y_val,
        X_test if has_test else None, y_test if has_test else None,
        patient_tag=f'{patient_tag}_coral_lambda{args.coral_lambda}',
        model_version=args.model_version,
    )
    # patient_tag passed UNMODIFIED here (not the coral-suffixed variant) so
    # _write_ckpt_manifest resolves the correct multi_scaler.json path.
    _write_ckpt_manifest(
        deploy_ckpt, patient_tag=patient_tag,
        seed=args.seed, finetune_from=None, gradual_unfreeze=False,
        model_version=args.model_version,
    )
    print(f"\n[CORAL] Done. Next -- convert + eval on each held-out patient:")
    print(f"  python3 src/models/convert_to_snn.py --patient multi --base multi "
          f"--variant coral_lambda{args.coral_lambda} --eval-patient chb03 "
          f"--model-version {args.model_version} --cal-samples 1024")
    print(f"  # repeat --eval-patient for chb10 chb13 chb15 chb16 chb20")
    print(f"  python3 src/evaluation/eval_event_level.py --fbz "
          f"results/seizure_model_multi_coral_lambda{args.coral_lambda}"
          f"_v{args.model_version}_w4a4.fbz --eval-patient chb03  # repeat per patient")
    sys.exit(0)

if args.finetune_from:
    # ── Fine-tuning path ──────────────────────────────────────────────
    base_ckpt = f'results/best_ann_{args.finetune_from}_v{args.model_version}.h5'
    if not os.path.exists(base_ckpt):
        sys.exit(
            f"ERROR: base checkpoint not found: {base_ckpt}\n"
            "Train the base model first (or check the path)."
        )
    print(f"\nLoading base model: {base_ckpt}")
    base_model = keras.models.load_model(base_ckpt)
    print(f"\nLoading base model: {base_ckpt}")
    base_model = keras.models.load_model(base_ckpt)

    # ── Gate 2c: personalised training must scale to the target patient's
    # OWN distribution, never the pool's — explicit check, not implicit ───
    if args.longctx:
        expected_scaler_path = (
            f'data/processed/multi_scaler_longctx_w{args.window_samples}.json'
            if args.multi_patient
            else f'data/processed/{args.patient}_scaler_longctx_w{args.window_samples}.json'
        )
    else:
        expected_scaler_path = ('data/processed/multi_scaler.json' if args.multi_patient
                                else f'data/processed/{args.patient}_scaler.json')
    scaler_owner = 'the pool' if args.multi_patient else "the patient's own"
    data_manifest_path = data_path + '.manifest.json'
    if os.path.exists(data_manifest_path):
        with open(data_manifest_path) as f:
            data_manifest = json.load(f)
        if data_manifest.get('scaler_path') != expected_scaler_path:
            sys.exit(
                f"ERROR: {data_path}'s manifest records scaler_path="
                f"{data_manifest.get('scaler_path')}, expected "
                f"{expected_scaler_path}. Fine-tuning data must be scaled "
                f"under {scaler_owner} scaler (Gate 2c) — refusing."
            )
        print(f"[Gate 2c] Confirmed: training data scaled under "
              f"{expected_scaler_path} ({scaler_owner} scaler).")
    else:
        print(f"[Gate 2c] WARNING: no manifest at {data_manifest_path} — "
              "cannot verify scaler provenance. Regenerate via "
              "build_dataset_multi.py / build_dataset.py to get this check.")

    if args.freeze_depth is not None:
        print(f"\n=== FREEZE-DEPTH FINE-TUNE (Gate 2) ===")
        print(f"Base  : {base_ckpt}")
        print(f"Target: {args.patient}")
        _run_freeze_depth_finetune(
            base_model, X_train, y_train, X_val, y_val,
            X_test=X_test if has_test else None,
            y_test=y_test if has_test else None,
            patient_tag=patient_tag,
            model_version=args.model_version,
            freeze_depth=args.freeze_depth,
            class_weight=args.class_weight,
            batch=args.batch,
            epochs=args.epochs,
            lr=1e-4,
        )
        _write_ckpt_manifest(
            f'results/best_ann_{patient_tag}_v{args.model_version}.h5',
            patient_tag=patient_tag, seed=args.seed,
            finetune_from=args.finetune_from, gradual_unfreeze=False,
            model_version=args.model_version,
            freeze_depth=args.freeze_depth,
        )
        sys.exit(0)

    elif args.gradual_unfreeze:
        # ... existing 3-phase block, unchanged except the BN patch below
        # ── Gradual unfreezing: Phase 2c adaptation for chb03 ────────
        #
        # Rationale: conv1 (9,7 spatio-temporal) captures patient-invariant
        # electrode co-activation patterns — keep frozen throughout.
        # conv2/conv3 refine ictal morphology — unfreeze progressively.
        #
        # Phase 1: Dense head only       — 20 epochs,  lr=1e-4
        # Phase 2: + conv3 unfrozen      — 10 epochs,  lr=5e-5
        # Phase 3: + conv2+conv3 unfrozen — 10 epochs, lr=1e-5
        print("\n=== GRADUAL UNFREEZING (Phase 2c) ===")
        print(f"Base  : {base_ckpt}")
        print(f"Target: {args.patient}")
        _run_gradual_unfreeze(
            base_model, X_train, y_train, X_val, y_val,
            X_test=X_test if has_test else None,
            y_test=y_test if has_test else None,
            patient_tag=args.patient,
            model_version=args.model_version,
            class_weight=args.class_weight,
            batch=args.batch,
        )
        # _run_gradual_unfreeze saves the best checkpoint and prints metrics — done.
        _write_ckpt_manifest(
            f'results/best_ann_{args.patient}_v{args.model_version}.h5',
            patient_tag=args.patient, seed=args.seed,
            finetune_from=args.finetune_from, gradual_unfreeze=True,
            model_version=args.model_version,
        )
        sys.exit(0)

    else:
        # ── Standard head-only fine-tuning (Phase 2a behaviour) ──────
        if args.model_version == 2:
            model = build_patient_adapted_model(base_model,
                                                freeze_until='relu3')
        else:
            # v1: freeze all conv blocks (layers up to last Dense)
            for layer in base_model.layers[:-4]:
                layer.trainable = False
            model = base_model

        print(f"\nFine-tuning (head-only) for {args.patient}")

elif args.init_from_ssl:
    # ── Candidate C: pretrained-trunk init (sec5d) ───────────────────────
    if not os.path.exists(args.init_from_ssl):
        sys.exit(f"ERROR: SSL-pretrained checkpoint not found: "
                 f"{args.init_from_ssl}\nRun pretrain_ssl.py first.")
    model = keras.models.load_model(args.init_from_ssl, compile=False)
    print(f"\nLoaded SSL-pretrained trunk init: {args.init_from_ssl}")
    print("(Candidate C, sec5d -- trunk pretrained via masked-window "
          "reconstruction on chb01/02/05 non-seizure windows only; Dense "
          "head still at random init. Normal multi-patient supervised "
          "training proceeds unchanged from here -- this is an "
          "initialisation change only, no architecture or training-loop "
          "difference from the C1 baseline it will be compared against.)")
else:
    # ── Fresh training ─────────────────────────────────────────────────
    model = build_fn()
    print(f"\nBuilding new model (v{args.model_version})")

# ── AKD1000 v1 compatibility check ────────────────────────────────────────────
try:
    with set_akida_version(AkidaVersion.v1):
        check_model_compatibility(model)
    print("AKD1000 v1 compatible ✓")
except Exception as e:
    print(f"WARNING: compatibility check raised: {e}")
    print("(check_model_compatibility returns None in cnn2snn 2.19.1 — "
          "this may be a false alarm)")

model.summary(print_fn=lambda x: print(f"  {x}"))

# ── 3. Single-phase training (multi-patient base or head-only fine-tune) ──────
ckpt_path = f'results/best_ann_{patient_tag}_v{args.model_version}.h5'

_compile_and_fit(
    model, X_train, y_train, X_val, y_val,
    epochs=args.epochs,
    lr=1e-4 if args.finetune_from else 1e-3,
    batch=args.batch,
    class_weight=args.class_weight,
    ckpt_path=ckpt_path,
)

# ── 4. Final evaluation ───────────────────────────────────────────────────────
model = keras.models.load_model(ckpt_path)
_evaluate_and_save(
    model, X_train, y_train, X_val, y_val,
    X_test if has_test else None, y_test if has_test else None,
    patient_tag=patient_tag,
    model_version=args.model_version,
)
_write_ckpt_manifest(
    ckpt_path, patient_tag=patient_tag, seed=args.seed,
    finetune_from=args.finetune_from, gradual_unfreeze=False,
    model_version=args.model_version,
)
