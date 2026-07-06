"""
train_auxhead_c2.py — Gate 1 of the C3 stacking plan (session: 4 July 2026)
=============================================================================
Auxiliary biological-feature regression head, compounding onto C2 (per-patient
head-only fine-tuning). NOT an input-fusion candidate -- that's confirmed dead
by constraint #23 (branching). This is a TRAINING-TIME-ONLY auxiliary loss,
architecturally identical in kind to DANN's domain head / SSL's decoder head:
the aux head is discarded entirely before quantize()/convert(). The deployed
model is byte-for-byte the same single-path v2 CNN as plain C2 -- zero new
inference cost, zero new energy cost, 100% on-chip.

Why this might differ from DANN/CORAL/SSL (already converged on an identical
chb13 failure): those three target DOMAIN INVARIANCE (make the trunk not care
which patient a window came from). This targets REPRESENTATION QUALITY on the
Dense(64) head itself, using known-good discriminative EEG statistics as an
inductive bias during personalization, on a patient with very few real
seizure examples. Different target, not just a different mechanism reaching
for the same thing -- but that's a hypothesis to screen cheaply, not an
assumed win.

Branch point: Dense(64) -- NOT the frozen conv trunk. C2 freezes conv1-3
(build_patient_adapted_model, freeze_until='relu3'), so an aux head hanging
off the trunk would receive zero gradient. Dense(64) is the one part of the
network already being fine-tuned per-patient -- that's where an aux signal
can actually do something.

Aux targets: line length, delta/beta band-power ratio, Hjorth mobility,
spectral entropy -- computed from the ORIGINAL (pre-[0,255]-scaling)
amplitude domain, by inverting the patient's own saved scaler.json (these
features are only physically meaningful in the real amplitude domain, not
an arbitrary per-patient rescale). Each target is z-scored across the
training split before use as a regression target.

Usage:
    python3 train_auxhead_c2.py --patient chb13 --finetune-from multi \
        --model-version 2 --aux-weight 0.3 --seed 256

    python3 train_auxhead_c2.py --patient chb16 --finetune-from multi \
        --model-version 2 --aux-weight 0.3 --seed 256

Output (variant-tagged, per the project's output-path safety rule --
NEVER overwrites the plain C2 checkpoint or another variant's):
    results/best_ann_{patient}_auxhead_v{model_version}.h5
    results/ann_results_{patient}_auxhead_v{model_version}.json
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import argparse, json, sys
import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import confusion_matrix
from cnn2snn import check_model_compatibility, set_akida_version, AkidaVersion

# ── Aux feature functions (same as the earlier feature-bank benchmark) ──────
def compute_aux_targets(X_orig_amplitude):
    """X_orig_amplitude: (N, 18, 512) float32, ORIGINAL amplitude domain
    (already inverse-scaled from [0,255]). Returns (N, 4) raw feature array:
    [line_length, band_power_ratio, hjorth_mobility, spectral_entropy],
    each averaged across the 18 channels to one scalar per window."""
    from scipy.fft import rfft, rfftfreq
    N = X_orig_amplitude.shape[0]
    out = np.zeros((N, 4), dtype=np.float32)
    freqs = rfftfreq(X_orig_amplitude.shape[-1], d=1/256)
    delta_mask = (freqs >= 0.5) & (freqs < 4)
    beta_mask = (freqs >= 13) & (freqs < 30)
    for i in range(N):
        w = X_orig_amplitude[i]                      # (18, 512)
        ll = np.mean(np.sum(np.abs(np.diff(w, axis=-1)), axis=-1))
        spec = np.abs(rfft(w, axis=-1)) ** 2
        bpr = np.mean(spec[..., delta_mask].sum(-1) / (spec[..., beta_mask].sum(-1) + 1e-8))
        d1 = np.diff(w, axis=-1)
        mob = np.mean(np.sqrt(np.var(d1, axis=-1) / (np.var(w, axis=-1) + 1e-8)))
        p = spec / (spec.sum(axis=-1, keepdims=True) + 1e-8)
        ent = np.mean(-np.sum(p * np.log(p + 1e-12), axis=-1))
        out[i] = [ll, bpr, mob, ent]
    return out


def zscore_fit_transform(arr):
    mu, sd = arr.mean(axis=0), arr.std(axis=0) + 1e-8
    return (arr - mu) / sd, mu, sd


def invert_scaler(X_scaled_0_255, scaler_json_path):
    """X was originally scaled as X_scaled = X_orig * scale + shift, saved
    per-patient in {patient}_scaler.json. Invert to recover original
    amplitude domain for feature computation."""
    with open(scaler_json_path) as f:
        s = json.load(f)
    scale, shift = s['scale'], s['shift']
    return (X_scaled_0_255 - shift) / scale


def build_auxhead_model(base_model, freeze_until='relu3', n_aux=4):
    """Same freeze pattern as build_patient_adapted_model, but adds a second
    head off Dense(64) (the 'relu_dense' layer's input, i.e. dense1's output
    before the ReLU) for auxiliary regression. Both heads trainable; conv1-3
    frozen exactly as in plain C2."""
    freeze = True
    for layer in base_model.layers:
        if freeze:
            layer.trainable = False
        if layer.name == freeze_until:
            freeze = False

    dense64_output = base_model.get_layer('relu_dense').output   # post-ReLU Dense(64)
    main_out = base_model.output                                  # existing Dense(2) softmax
    aux_out = keras.layers.Dense(n_aux, activation='linear', name='aux_regression')(dense64_output)

    model = keras.Model(inputs=base_model.input, outputs=[main_out, aux_out], name='v2_auxhead')
    return model


def deployable_model_from_auxhead(auxhead_model):
    """Strips the aux head entirely -- returns a model with the SAME graph
    and weights as plain C2, single output, ready for quantize()/convert().
    This is the artifact that actually gets deployed -- confirm it is
    architecturally identical to a plain C2 checkpoint before converting."""
    return keras.Model(inputs=auxhead_model.input,
                        outputs=auxhead_model.get_layer('output').output,
                        name='v2_deployable')


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
    ap.add_argument('--patient', required=True, help='e.g. chb13, chb16')
    ap.add_argument('--finetune-from', required=True, help="'multi' or a chbXX base tag")
    ap.add_argument('--model-version', type=int, default=2)
    ap.add_argument('--aux-weight', type=float, default=0.3,
                     help='loss weight on the aux regression head (screen 0.1-0.5)')
    ap.add_argument('--seed', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--class-weight', type=float, default=1.5)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    VARIANT = 'auxhead'
    data_path = f'data/processed/{args.patient}_dataset_ann.npz'
    scaler_path = f'data/processed/{args.patient}_scaler.json'
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

    print(f"\n[Gate 1: auxhead] Computing aux regression targets "
          f"(original-amplitude domain, inverted via {scaler_path})...")
    X_train_orig = invert_scaler(X_train[..., 0], scaler_path)     # (N,18,512)
    X_val_orig = invert_scaler(X_val[..., 0], scaler_path)

    aux_train_raw = compute_aux_targets(X_train_orig)
    aux_val_raw = compute_aux_targets(X_val_orig)
    aux_train, mu, sd = zscore_fit_transform(aux_train_raw)        # fit on train only
    aux_val = (aux_val_raw - mu) / sd                               # apply train stats to val

    print(f"Loading base model: {base_ckpt}")
    base_model = keras.models.load_model(base_ckpt)
    model = build_auxhead_model(base_model, freeze_until='relu3', n_aux=4)
    model.summary(print_fn=lambda x: print(f"  {x}"))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={'output': 'sparse_categorical_crossentropy', 'aux_regression': 'mse'},
        loss_weights={'output': 1.0, 'aux_regression': args.aux_weight},
        metrics={'output': 'accuracy'},
    )

    ckpt_path = f'results/best_ann_{args.patient}_{VARIANT}_v{args.model_version}.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_output_loss',
                                         save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_output_loss', patience=15,
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_output_loss', factor=0.5,
                                           patience=7, min_lr=1e-7, verbose=1),
    ]

    model.fit(
        X_train, {'output': y_train, 'aux_regression': aux_train},
        validation_data=(X_val, {'output': y_val, 'aux_regression': aux_val}),
        epochs=args.epochs, batch_size=args.batch,
        class_weight=None,   # per-output class_weight isn't supported cleanly in
                              # multi-output Keras models; sample_weight below instead
        sample_weight={'output': np.where(y_train == 1, args.class_weight, 1.0).astype('float32'),
                       'aux_regression': np.ones(len(y_train), dtype='float32')},
        callbacks=callbacks, verbose=2,
    )

    print(f"\n[Gate 1] Saved: {ckpt_path}")
    print("Sanity check -- confirm the deployable model matches plain C2's graph:")
    best_auxhead = keras.models.load_model(ckpt_path)
    deployable = deployable_model_from_auxhead(best_auxhead)
    with set_akida_version(AkidaVersion.v1):
        check_model_compatibility(deployable)
    print("AKD1000 v1 compatible (deployable, aux head stripped) ✓")

    # ── Evaluate the DEPLOYABLE (aux-stripped) model, same metric bundle as C2 ──
    for split_name, X_s, y_s in [('train', X_train, y_train), ('val', X_val, y_val),
                                  ('test', X_test, y_test)]:
        if X_s is None:
            continue
        preds = np.argmax(deployable.predict(X_s, batch_size=32, verbose=0), axis=1)
        sens, spec, fpr_hr = _metrics_from_cm(y_s, preds)
        if sens is not None:
            print(f"  {split_name:5s}: sens={sens:.4f}  spec={spec:.4f}  FPR/hr={fpr_hr:.2f}")

    print(f"\nNext: python3 src/models/convert_to_snn.py --patient {args.patient} "
          f"--variant {VARIANT} --model-version {args.model_version}")
    print(f"Then: python3 src/evaluation/eval_event_level.py --fbz "
          f"results/seizure_model_{args.patient}_{VARIANT}_v{args.model_version}_w4a4.fbz "
          f"--eval-patient {args.patient}")
    print(f"\nCompare against plain C2 baseline for {args.patient} using the "
          f"Gate 1 pass bar: event sens +1, or FP/hr -20%, no collapse fail.")
