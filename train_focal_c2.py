"""
train_focal_c2.py — Gate 2 of the C3 stacking plan (session: 4 July 2026)
=============================================================================
Focal loss instead of flat class-weighted crossentropy, on the exact same
C2 head-only personalization fine-tune. Gate 1 (auxiliary regression head)
closed as an honest negative -- see results/event_results_*_auxhead_*.json.

Rationale: the current class-weight=1.5 flat multiplier treats every
seizure window as equally important to get right. Focal loss instead
down-weights windows the model ALREADY classifies confidently (correct or
not) and concentrates gradient on the ones it's still unsure about -- which
is a more targeted fit for chb13/chb16's specific failure modes (chb13:
some seizure windows easy, some genuinely ambiguous morphology; chb16:
flat data-efficiency curve suggests the model isn't extracting much signal
from the few real examples it has, focal loss may help it spend more
effort on the informative ones).

No architecture change at all -- same build_patient_adapted_model, same
frozen conv1-3, same single-output model. This is a training-time-only
change and needs zero sanity-check for dual outputs (unlike Gate 1).

Usage:
    python3 train_focal_c2.py --patient chb13 --finetune-from multi \
        --model-version 2 --gamma 2.0 --alpha 0.75 --seed 256

    python3 train_focal_c2.py --patient chb16 --finetune-from multi \
        --model-version 2 --gamma 2.0 --alpha 0.75 --seed 256

Output (variant-tagged -- matches convert_to_snn.py's expected
{patient}_v{version}_{variant} ordering, confirmed against Gate 1's bug):
    results/best_ann_{patient}_v{model_version}_focal.h5
    results/ann_results_{patient}_v{model_version}_focal.json
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import argparse, json, sys
import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import confusion_matrix
from cnn2snn import check_model_compatibility, set_akida_version, AkidaVersion


def build_patient_adapted_model(base_model, freeze_until='relu3'):
    """Same freeze pattern as the project's existing plain-C2 function --
    reproduced here so this script is self-contained and doesn't depend on
    importing from train_baseline.py."""
    freeze = True
    for layer in base_model.layers:
        if freeze:
            layer.trainable = False
        if layer.name == freeze_until:
            freeze = False
    return base_model


def sparse_focal_loss(gamma=2.0, alpha=0.75):
    """Binary focal loss for sparse integer labels (0/1) against a 2-unit
    softmax output. alpha weights the positive (seizure) class -- analogous
    to class_weight but multiplicative on the focal term, not a separate
    sample_weight. gamma controls how strongly confident-correct examples
    are down-weighted (gamma=0 reduces to plain weighted crossentropy)."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.gather(y_pred, y_true, batch_dims=1, axis=1) if False else \
             tf.reduce_sum(y_pred * tf.one_hot(y_true, 2), axis=1)
        ce = -tf.math.log(pt)
        focal_weight = tf.pow(1.0 - pt, gamma)
        alpha_factor = tf.where(tf.equal(y_true, 1),
                                 tf.fill(tf.shape(y_true), alpha),
                                 tf.fill(tf.shape(y_true), 1.0 - alpha))
        alpha_factor = tf.cast(alpha_factor, tf.float32)
        return tf.reduce_mean(alpha_factor * focal_weight * ce)
    return loss_fn


def _metrics_from_cm(y_true, y_pred):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None, None, None
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return None, None, None
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    n_neg = tn + fp
    fpr_hr = fp / (n_neg * 2 / 3600) if n_neg > 0 else 0.0
    return float(sens), float(spec), float(fpr_hr)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient', required=True)
    ap.add_argument('--finetune-from', required=True)
    ap.add_argument('--model-version', type=int, default=2)
    ap.add_argument('--gamma', type=float, default=2.0,
                     help='focal focusing param -- screen 1.0-3.0')
    ap.add_argument('--alpha', type=float, default=0.75,
                     help='seizure-class weight, 0-1 -- screen 0.5-0.9')
    ap.add_argument('--seed', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    VARIANT = 'focal'
    data_path = f'data/processed/{args.patient}_dataset_ann.npz'
    base_ckpt = f'results/best_ann_{args.finetune_from}_v{args.model_version}.h5'

    if not os.path.exists(data_path):
        sys.exit(f"ERROR: {data_path} not found.")
    if not os.path.exists(base_ckpt):
        sys.exit(f"ERROR: base checkpoint not found: {base_ckpt}")

    data = np.load(data_path)
    X_train = data['X_train'][..., np.newaxis].astype('float32')
    y_train = data['y_train']
    X_val = data['X_val'][..., np.newaxis].astype('float32')
    y_val = data['y_val']
    X_test = data['X_test'][..., np.newaxis].astype('float32') if 'X_test' in data.files else None
    y_test = data['y_test'] if 'y_test' in data.files else None

    print(f"Loading base model: {base_ckpt}")
    base_model = keras.models.load_model(base_ckpt)
    model = build_patient_adapted_model(base_model, freeze_until='relu3')
    model.summary(print_fn=lambda x: print(f"  {x}"))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=sparse_focal_loss(gamma=args.gamma, alpha=args.alpha),
        metrics=['accuracy'],
    )

    ckpt_path = f'results/best_ann_{args.patient}_v{args.model_version}_{VARIANT}.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss',
                                         save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=7, min_lr=1e-7, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch,
        callbacks=callbacks, verbose=2,
    )

    print(f"\n[Gate 2] Saved: {ckpt_path}")
    model = keras.models.load_model(ckpt_path, compile=False)
    with set_akida_version(AkidaVersion.v1):
        check_model_compatibility(model)
    print("AKD1000 v1 compatible ✓ (single-output, no stripping needed)")

    for split_name, X_s, y_s in [('train', X_train, y_train), ('val', X_val, y_val),
                                  ('test', X_test, y_test)]:
        if X_s is None:
            continue
        preds = np.argmax(model.predict(X_s, batch_size=32, verbose=0), axis=1)
        sens, spec, fpr_hr = _metrics_from_cm(y_s, preds)
        if sens is not None:
            print(f"  {split_name:5s}: sens={sens:.4f}  spec={spec:.4f}  FPR/hr={fpr_hr:.2f}")

    print(f"\nNext: python3 src/models/convert_to_snn.py --patient {args.patient} "
          f"--variant {VARIANT} --model-version {args.model_version}")
    print(f"Then: python3 src/evaluation/eval_event_level.py --fbz "
          f"results/seizure_model_{args.patient}_{VARIANT}_v{args.model_version}_w4a4.fbz "
          f"--eval-patient {args.patient}")
    print(f"\nCompare against plain C2 baseline using the same Gate bar: "
          f"event sens +1, or FP/hr -20%, no collapse fail.")
