"""
train_distill_c2.py — Gate 3 of the C3 stacking plan (session: 4 July 2026)
=============================================================================
Knowledge distillation from a multi-seed teacher ensemble into ONE deployed
model. Gates 1 (aux regression head) and 2 (focal loss) both closed as
honest negatives, converging on the identical 3/6 chb13 events at nearly
identical latencies -- a stable personalisation ceiling, not something either
training-time change could move. This is a genuinely different mechanism:
it directly targets the documented seed-sensitivity finding (identical
architecture/data, different seed, event sensitivity swings 0.000-1.000) --
if the "real" signal differs meaningfully across seeds, an ensemble's soft
labels carry more information than any single seed's hard labels, and
distilling that into one model costs NOTHING extra at deployment (student is
the same single-path v2 CNN -- teachers are training-time only, discarded
after producing soft labels, same discard-pattern as everything else this
session).

TWO-STEP usage:

Step 1 -- train N teacher seeds (head-only C2 fine-tune, identical to plain
C2 except tagged by seed):
    python3 train_distill_c2.py --mode teachers --patient chb13 \
        --finetune-from multi --model-version 2 --seeds 111 222 333

Step 2 -- distil the teacher ensemble's soft labels into one student:
    python3 train_distill_c2.py --mode distill --patient chb13 \
        --finetune-from multi --model-version 2 --seeds 111 222 333 \
        --temperature 3.0 --kd-alpha 0.5 --student-seed 256

Output:
    results/best_ann_{patient}_v{model_version}_teacher_seed{seed}.h5   (step 1, per seed)
    results/best_ann_{patient}_v{model_version}_distill.h5              (step 2, the ONE deployed model)
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
    freeze = True
    for layer in base_model.layers:
        if freeze:
            layer.trainable = False
        if layer.name == freeze_until:
            freeze = False
    return base_model


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


def load_data(patient):
    data_path = f'data/processed/{patient}_dataset_ann.npz'
    if not os.path.exists(data_path):
        sys.exit(f"ERROR: {data_path} not found.")
    data = np.load(data_path)
    X_train = data['X_train'][..., np.newaxis].astype('float32')
    y_train = data['y_train']
    X_val = data['X_val'][..., np.newaxis].astype('float32')
    y_val = data['y_val']
    X_test = data['X_test'][..., np.newaxis].astype('float32') if 'X_test' in data.files else None
    y_test = data['y_test'] if 'y_test' in data.files else None
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_one_teacher(patient, base_ckpt, model_version, seed, epochs, batch, class_weight):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    X_train, y_train, X_val, y_val, _, _ = load_data(patient)
    base_model = keras.models.load_model(base_ckpt)
    model = build_patient_adapted_model(base_model, freeze_until='relu3')
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ckpt_path = f'results/best_ann_{patient}_v{model_version}_teacher_seed{seed}.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=0),
    ]
    print(f"\n=== Training teacher seed {seed} ===")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch,
              class_weight={0: 1.0, 1: class_weight},
              callbacks=callbacks, verbose=2)
    print(f"Saved teacher: {ckpt_path}")
    return ckpt_path


def kd_loss_fn(temperature=3.0, kd_alpha=0.5):
    """Combined loss: (1-kd_alpha)*hard-label CE + kd_alpha*T^2*KL(teacher||student)
    at softened temperature T. y_true packs [hard_label, teacher_prob_0, teacher_prob_1]
    column-wise so a single label tensor carries both signals through .fit()."""
    def loss_fn(y_true, y_pred):
        hard_label = tf.cast(y_true[:, 0], tf.int32)
        teacher_probs = y_true[:, 1:3]
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        hard_ce = tf.keras.losses.sparse_categorical_crossentropy(hard_label, y_pred)

        # soften both distributions at temperature T (teacher already softmax
        # probs, so re-derive logits implicitly via power-law softening)
        student_soft = tf.pow(y_pred, 1.0 / temperature)
        student_soft = student_soft / tf.reduce_sum(student_soft, axis=1, keepdims=True)
        teacher_soft = tf.pow(teacher_probs, 1.0 / temperature)
        teacher_soft = teacher_soft / tf.reduce_sum(teacher_soft, axis=1, keepdims=True)
        kd = tf.reduce_sum(teacher_soft * (tf.math.log(teacher_soft + 1e-8) -
                                            tf.math.log(student_soft + 1e-8)), axis=1)

        return (1.0 - kd_alpha) * hard_ce + kd_alpha * (temperature ** 2) * kd
    return loss_fn


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['teachers', 'distill'])
    ap.add_argument('--patient', required=True)
    ap.add_argument('--finetune-from', required=True)
    ap.add_argument('--model-version', type=int, default=2)
    ap.add_argument('--seeds', type=int, nargs='+', required=True,
                     help='teacher seeds, e.g. 111 222 333')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--class-weight', type=float, default=1.5)
    ap.add_argument('--temperature', type=float, default=3.0)
    ap.add_argument('--kd-alpha', type=float, default=0.5)
    ap.add_argument('--student-seed', type=int, default=256)
    args = ap.parse_args()

    base_ckpt = f'results/best_ann_{args.finetune_from}_v{args.model_version}.h5'
    if not os.path.exists(base_ckpt):
        sys.exit(f"ERROR: base checkpoint not found: {base_ckpt}")

    if args.mode == 'teachers':
        for seed in args.seeds:
            train_one_teacher(args.patient, base_ckpt, args.model_version, seed,
                               args.epochs, args.batch, args.class_weight)
        print(f"\n[Gate 3 step 1] {len(args.seeds)} teachers trained. Next:")
        print(f"  python3 train_distill_c2.py --mode distill --patient {args.patient} "
              f"--finetune-from {args.finetune_from} --model-version {args.model_version} "
              f"--seeds {' '.join(map(str, args.seeds))} --temperature {args.temperature} "
              f"--kd-alpha {args.kd_alpha} --student-seed {args.student_seed}")
        sys.exit(0)

    # ── distill mode ──────────────────────────────────────────────────────
    VARIANT = 'distill'
    teacher_ckpts = [f'results/best_ann_{args.patient}_v{args.model_version}_teacher_seed{s}.h5'
                      for s in args.seeds]
    for tc in teacher_ckpts:
        if not os.path.exists(tc):
            sys.exit(f"ERROR: teacher checkpoint not found: {tc}\n"
                     f"Run --mode teachers first with the same --seeds.")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.patient)

    print(f"\n[Gate 3 step 2] Loading {len(teacher_ckpts)} teachers, "
          f"computing ensemble soft labels on the training split...")
    teacher_probs_sum = np.zeros((len(X_train), 2), dtype=np.float32)
    for tc in teacher_ckpts:
        teacher = keras.models.load_model(tc, compile=False)
        teacher_probs_sum += teacher.predict(X_train, batch_size=32, verbose=0)
    teacher_probs = teacher_probs_sum / len(teacher_ckpts)

    # pack [hard_label, teacher_prob_0, teacher_prob_1] into one label tensor
    y_train_packed = np.concatenate(
        [y_train.reshape(-1, 1).astype('float32'), teacher_probs], axis=1)

    # for validation, use hard labels only + a placeholder teacher column
    # (val loss only needs the hard-CE term to be meaningful for early
    # stopping; we don't have cheap held-out teacher soft labels and don't
    # need them for checkpoint selection)
    val_teacher_placeholder = np.tile(np.array([0.5, 0.5], dtype='float32'), (len(y_val), 1))
    y_val_packed = np.concatenate(
        [y_val.reshape(-1, 1).astype('float32'), val_teacher_placeholder], axis=1)

    tf.random.set_seed(args.student_seed)
    np.random.seed(args.student_seed)
    base_model = keras.models.load_model(base_ckpt)
    student = build_patient_adapted_model(base_model, freeze_until='relu3')
    student.compile(optimizer=keras.optimizers.Adam(1e-4),
                     loss=kd_loss_fn(args.temperature, args.kd_alpha))

    ckpt_path = f'results/best_ann_{args.patient}_v{args.model_version}_{VARIANT}.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
    ]
    student.fit(X_train, y_train_packed, validation_data=(X_val, y_val_packed),
                epochs=args.epochs, batch_size=args.batch, callbacks=callbacks, verbose=2)

    print(f"\n[Gate 3] Saved: {ckpt_path}")
    student = keras.models.load_model(ckpt_path, compile=False)
    with set_akida_version(AkidaVersion.v1):
        check_model_compatibility(student)
    print("AKD1000 v1 compatible ✓")

    # re-save with a plain loss -- same fix as Gate 2, the custom kd_loss_fn
    # closure would otherwise break convert_to_snn.py's plain load_model()
    student.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    student.save(ckpt_path)
    print(f"Re-saved {ckpt_path} with a plain serializable loss")

    for split_name, X_s, y_s in [('train', X_train, y_train), ('val', X_val, y_val),
                                  ('test', X_test, y_test)]:
        if X_s is None:
            continue
        preds = np.argmax(student.predict(X_s, batch_size=32, verbose=0), axis=1)
        sens, spec, fpr_hr = _metrics_from_cm(y_s, preds)
        if sens is not None:
            print(f"  {split_name:5s}: sens={sens:.4f}  spec={spec:.4f}  FPR/hr={fpr_hr:.2f}")

    print(f"\nNext: python3 src/models/convert_to_snn.py --patient {args.patient} "
          f"--variant {VARIANT} --model-version {args.model_version}")
    print(f"Then: python3 src/evaluation/eval_event_level.py --fbz "
          f"results/seizure_model_{args.patient}_{VARIANT}_v{args.model_version}_w4a4.fbz "
          f"--eval-patient {args.patient}")
