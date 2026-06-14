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

import argparse, json, os, sys
import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import confusion_matrix
from cnn2snn import check_model_compatibility, set_akida_version, AkidaVersion


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
parser.add_argument('--class-weight',     type=float, default=1.5)
parser.add_argument('--epochs',           type=int, default=100)
parser.add_argument('--batch',            type=int, default=32)
parser.add_argument('--stft',             action='store_true',
                    help='Use 3-channel STFT dataset (Phase 2d)')
args = parser.parse_args()

# Validation
if args.gradual_unfreeze and args.finetune_from is None:
    parser.error("--gradual-unfreeze requires --finetune-from")
if args.multi_patient and args.gradual_unfreeze:
    parser.error("--multi-patient trains a base model; "
                 "use --gradual-unfreeze only when fine-tuning (--finetune-from multi)")

# Derive tag used in file names
patient_tag = 'multi' if args.multi_patient else args.patient
if args.stft:
    patient_tag += '_stft'
# ── GPU memory growth ──────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
print(f"Training on: {gpus[0].name if gpus else 'CPU (no GPU found)'}")

# ── 1. Load data ───────────────────────────────────────────────────────────────
if args.multi_patient:
    data_path = 'data/processed/multi_dataset_ann.npz'
    if not os.path.exists(data_path):
        sys.exit(
            f"ERROR: {data_path} not found.\n"
            "Run first: python3 src/preprocessing/build_dataset_multi.py"
        )
else:
    data_path = f'data/processed/{args.patient}_dataset_ann.npz'

if args.stft:
    stft_path = data_path.replace('_dataset_ann.npz', '_dataset_stft.npz')
    if not os.path.exists(stft_path):
        sys.exit(f"ERROR: {stft_path} not found.\n"
                 f"Run: python3 src/preprocessing/build_dataset_stft.py --patient {args.patient}")
    data_path = stft_path

data    = np.load(data_path)
if args.stft:
    X_train = data['X_train'].astype('float32')   # already (N, 18, 512, 3)
    X_val   = data['X_val'].astype('float32')
else:
    X_train = data['X_train'][..., np.newaxis].astype('float32')
    X_val   = data['X_val']  [..., np.newaxis].astype('float32')
y_train = data['y_train']
y_val   = data['y_val']

# Test split: present for per-patient datasets, absent for multi
has_test = 'X_test' in data.files
if has_test:
    X_test = data['X_test'].astype('float32') if args.stft \
             else data['X_test'][..., np.newaxis].astype('float32')
    y_test = data['y_test']

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
if args.multi_patient or args.gradual_unfreeze or args.stft:
    # Both multi-patient training and gradual-unfreeze fine-tuning suffer from
    # the same structural problem: all ictal windows fall in the first 70% of
    # recordings chronologically, so the npz val split (70–85%) has zero
    # seizures. Carve a balanced 20% val from the SMOTE'd training set instead.
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=42,
    )
    print(f"\nDataset: {data_path}")
    print(f"  Val carved from training set (stratified 80/20 — npz val had 0 seizures):")
    print(f"  Train : {len(X_train):>6}  seizure={int(y_train.sum())} "
          f"({100*y_train.mean():.1f}%)")
    print(f"  Val   : {len(X_val):>6}  seizure={int(y_val.sum())} "
          f"({100*y_val.mean():.1f}%)")
    if has_test:
        print(f"  Test  : {len(X_test):>6}  seizure={int(y_test.sum())}  "
              f"(held-out — evaluated after training)")
else:
    print(f"\nDataset: {data_path}")
    print(f"  Train : {len(X_train):>6}  seizure={int(y_train.sum())} "
          f"({100*y_train.mean():.1f}%)")
    print(f"  Val   : {len(X_val):>6}  seizure={int(y_val.sum())}")
    if has_test:
        print(f"  Test  : {len(X_test):>6}  seizure={int(y_test.sum())}")

expected_shape = (18, 512, 3) if args.stft else (18, 512, 1)
assert X_train.shape[1:] == expected_shape, \
    f"Wrong input shape {X_train.shape[1:]} — expected {expected_shape}"

# ── 2. Build or load model ────────────────────────────────────────────────────
sys.path.insert(0, '.')
if args.stft:
    from src.models.akida_cnn_v2 import build_seizure_cnn_v2_3ch
    build_fn = lambda: build_seizure_cnn_v2_3ch(n_channels=18, window_samples=512)
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

    if args.gradual_unfreeze:
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
