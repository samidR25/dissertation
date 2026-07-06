#!/usr/bin/env python3
"""
apply_coral_first_experiment_patch.py
======================================
CORAL first-experiment scoping (Handoff_post_dann_scoping_to_implementation.md
sec3). A-i variant only: multi-source, pool-internal pairwise covariance
alignment (chb01<->chb02, chb01<->chb05, chb02<->chb05). No held-out patient
data touched -- A-ii (source->target with an unlabeled calibration window)
is explicitly NOT implemented here; that needs a separate framing decision
flagged to Dr. Pham first (sec3b).

Applies two edits:

  1. src/models/akida_cnn_v2.py
       - build_seizure_cnn_v2_coral()  -- IDENTICAL trunk + main head to
         build_seizure_cnn_v2 (same layer names -- extract_deployable_
         submodel() needs zero changes to work on this). Second output is
         the raw 'flatten' feature tensor (1536-dim, training-only).
       - make_coral_loss()  -- Keras-loss-shaped factory. y_true = per-
         sample domain label, y_pred = that sample's flatten features.
         Computes pairwise (all domain pairs present in the batch)
         normalised squared Frobenius distance between per-domain
         covariance matrices, Sun & Saenko (2016) style. lambda_coral is
         NOT baked in -- applied via compile()'s loss_weights, so the
         {0.01, 0.1} sweep (sec3d) is a CLI flag, not a rebuild.
       - coral_pairwise_distance()  -- same math, but a plain eager
         function (not a graph-mode loss) for the before/after covariance-
         distance readout sec3d's third bullet requires -- CORAL's
         equivalent of DANN's domain-accuracy control (DANN handoff sec3.2).
       - CoralDistanceMonitor  -- Keras Callback wrapping the above, logs
         the readout before training and after every epoch.

  2. src/models/train_baseline.py
       - --coral / --coral-lambda flags, mutually exclusive with --dann
         (run separately for direct comparability against the closed DANN
         table) and with --finetune-from/--stft/--longctx (first screen
         only, same restriction --dann already has).
       - domain-label loading block generalised from `if args.dann:` to
         `if args.dann or args.coral:` -- no behaviour change for --dann.
       - full CORAL training path: build two-output model -> compile with
         dict losses/loss_weights -> fit with CoralDistanceMonitor ->
         extract deployable submodel (reused from DANN, untouched) ->
         AKD1000 v1 compat check on the DEPLOYABLE model -> save + manifest.
       - requires a dataset with domain_train/domain_val (already present
         per this session's verification of build_dataset_multi.py's
         on-disk state -- see handoff sec7); hard-refuses otherwise.

Does NOT touch src/preprocessing/build_dataset_multi.py -- verified this
session to already carry the DANN session's domain-labelling patch.

Run from ~/dissertation/ with akida_env activated:
    python3 apply_coral_first_experiment_patch.py

Hard-refuses (exits nonzero, writes nothing to the file it fails on) if
any anchor text isn't found exactly once -- re-run after checking whether
the file changed since this patch was written against the 2 July 2026
snapshot (post-DANN-patch state).
"""
import sys


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


# ============================================================================
# 1. akida_cnn_v2.py
# ============================================================================
AKIDA_CNN_V2_PATH = 'src/models/akida_cnn_v2.py'

akida_old_1 = """def build_patient_adapted_model(base_model, freeze_until='relu3'):"""

akida_new_1 = '''def build_seizure_cnn_v2_coral(n_channels=18, window_samples=512):
    """
    CORAL scoping experiment, A-i multi-source variant (Handoff_post_dann_
    scoping_to_implementation.md sec3). Trunk + main head are IDENTICAL
    (same layer names) to build_seizure_cnn_v2 -- no architecture change
    vs the deployed graph, same attachment point DANN used (flattened
    bottleneck, 1536-dim), which keeps this genuinely comparable as an
    alternative mechanism at the same point rather than a confound of
    also moving where invariance is enforced (sec3c).

    Second output is the RAW 'flatten' feature tensor itself -- no extra
    trainable layer, purely a training-time readout used by
    make_coral_loss()/CoralDistanceMonitor below. Discarded entirely
    before quantize()/convert(), same "training-only auxiliary structure"
    pattern already established safe for DANN's domain head.
    """
    inp = keras.Input(shape=(n_channels, window_samples, 1), name='eeg_input')
    x = keras.layers.Rescaling(scale=1.0 / 255.0, name='rescaling')(inp)

    x = keras.layers.Conv2D(32, (9, 7), strides=(1, 4),
        padding='valid', use_bias=False, name='conv1')(x)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.MaxPooling2D((1, 2), padding='valid', name='pool1')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu1')(x)

    x = keras.layers.Conv2D(64, (3, 3),
        padding='same', use_bias=False, name='conv2')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu2')(x)

    x = keras.layers.Conv2D(32, (3, 3),
        padding='same', use_bias=False, name='conv3')(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu3')(x)

    shared = keras.layers.Flatten(name='flatten')(x)

    # -- Main task head (identical to build_seizure_cnn_v2) --------------
    h = keras.layers.Dense(64, use_bias=False, name='dense1')(shared)
    h = keras.layers.ReLU(max_value=6.0, name='relu_dense')(h)
    main_out = keras.layers.Dense(2, activation='softmax', name='output')(h)

    return keras.Model(inputs=inp, outputs=[main_out, shared],
                        name='seizure_cnn_v2_coral')


def _coral_domain_covariances(features, domain_labels, n_domains):
    """Shared helper: per-domain (centered feature) covariance matrices +
    per-domain sample counts, for whichever domains happen to be present
    in this batch/set. Domains absent (count<=1) get a zero covariance
    and are excluded from any pairwise sum downstream via the count check
    -- never silently treated as a real zero-distance match."""
    domain_labels = tf.cast(tf.reshape(domain_labels, [-1]), tf.int32)
    feats = tf.cast(features, tf.float32)
    covs, counts = [], []
    for dom in range(n_domains):
        mask = tf.equal(domain_labels, dom)
        fd = tf.boolean_mask(feats, mask)
        n = tf.cast(tf.shape(fd)[0], tf.float32)
        fdc = fd - tf.reduce_mean(fd, axis=0, keepdims=True)
        cov = tf.matmul(fdc, fdc, transpose_a=True) / tf.maximum(n - 1.0, 1.0)
        covs.append(cov)
        counts.append(n)
    return covs, counts


def make_coral_loss(n_domains=3):
    """
    Keras-loss-shaped factory for the CORAL covariance-alignment term
    (Sun & Saenko, 2016): mean over all domain pairs PRESENT in the batch
    of the normalised squared Frobenius distance between their covariance
    matrices, ||C_i - C_j||_F^2 / (4*d^2). Pairs where either domain has
    <=1 sample in this batch contribute 0, not a spurious zero-vs-zero
    match.

    Used as the loss for the model's second output (raw 'flatten'
    features). y_true = per-sample domain label (int, batch-shaped);
    y_pred = that sample's flatten-layer activations (batch, 1536).

    lambda_coral is intentionally NOT baked in here -- apply it via
    compile()'s loss_weights, so the {0.01, 0.1} sweep (sec3d) is a CLI
    flag, not a rebuild. CORAL losses sit on a very different scale than
    cross-entropy -- do not reuse DANN's lambda values.
    """
    def coral_loss(y_true, y_pred):
        d = tf.cast(tf.shape(y_pred)[1], tf.float32)
        covs, counts = _coral_domain_covariances(y_pred, y_true, n_domains)
        total = tf.constant(0.0, dtype=tf.float32)
        npairs = 0
        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                valid = tf.logical_and(counts[i] > 1.0, counts[j] > 1.0)
                pair_loss = tf.reduce_sum(tf.square(covs[i] - covs[j])) / (4.0 * d * d)
                total = total + tf.where(valid, pair_loss, 0.0)
                npairs += 1
        return total / float(npairs)
    return coral_loss


def coral_pairwise_distance(features, domain_labels, n_domains=3):
    """
    Raw (unweighted, lambda-free) CORAL covariance-distance metric -- the
    "did the alignment loss actually do something" readout (Handoff_post_
    dann_scoping_to_implementation.md sec3d, third bullet). Equivalent in
    spirit to DANN's before/after domain-accuracy control (DANN handoff
    sec3.2): report this BEFORE training (random-init trunk) and AFTER,
    on the same held-out val set, to confirm the loss term moved the
    thing it's supposed to move -- never assume it from the training
    curve alone.

    Plain eager function (not a graph-mode loss) -- call from a callback
    or standalone script, not inside the training loss itself.
    """
    d = float(tf.shape(features)[1])
    covs, counts = _coral_domain_covariances(features, domain_labels, n_domains)
    total, npairs = 0.0, 0
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            if float(counts[i]) > 1.0 and float(counts[j]) > 1.0:
                total += float(tf.reduce_sum(tf.square(covs[i] - covs[j]))) / (4.0 * d * d)
                npairs += 1
    return total / npairs if npairs else float('nan')


class CoralDistanceMonitor(keras.callbacks.Callback):
    """
    Wraps coral_pairwise_distance() as a Keras Callback: measures the raw
    covariance-distance on the (fixed) validation set before training and
    after every epoch, using the two-output model itself (main_out is
    discarded, only the 'flatten' output is used for the metric). This is
    the sec3d third-bullet requirement -- proof the alignment loss did
    something, not an assumption.
    """
    def __init__(self, coral_model, X_val, domain_val, n_domains):
        super().__init__()
        self.coral_model = coral_model
        self.X_val = X_val
        self.domain_val = domain_val
        self.n_domains = n_domains
        self.history = []

    def _measure(self):
        _, feats = self.coral_model.predict(self.X_val, verbose=0, batch_size=64)
        return coral_pairwise_distance(feats, self.domain_val, self.n_domains)

    def on_train_begin(self, logs=None):
        d0 = self._measure()
        self.history.append(('pre_training', d0))
        print(f"\\n[CORAL] Pre-training val covariance-distance: {d0:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        d = self._measure()
        self.history.append((f'epoch_{epoch}', d))
        print(f"[CORAL] Epoch {epoch} val covariance-distance: {d:.6f}")


def build_patient_adapted_model(base_model, freeze_until='relu3'):'''

# ============================================================================
# 2. train_baseline.py
# ============================================================================
TRAIN_BASELINE_PATH = 'src/models/train_baseline.py'

tb_old_1 = """parser.add_argument('--dann-lambda', type=float, default=0.1,
                    help='Domain-loss weight / GRL lambda for --dann '
                         '(default 0.1; first screen also tries 0.5)')
args = parser.parse_args()

if args.dann and not args.multi_patient:
    parser.error('--dann requires --multi-patient (domains = pool patients)')
if args.dann and (args.finetune_from or args.stft or args.longctx):
    parser.error('--dann is mutually exclusive with --finetune-from/--stft/--longctx '
                 '(first screen only)')"""

tb_new_1 = """parser.add_argument('--dann-lambda', type=float, default=0.1,
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
                 'DANN table')"""

tb_old_2 = """# ── DANN domain labels ──────────────────────────────────────────────────────────
if args.dann:
    if 'domain_train' not in data.files or 'domain_val' not in data.files:
        sys.exit(
            f"ERROR: {data_path} has no domain_train/domain_val arrays -- "
            "it predates the DANN-scoping patch to build_dataset_multi.py.\\n"
            "Rebuild via:\\n"
            "  python3 src/preprocessing/build_dataset_multi.py "
            "--patients chb01 chb02 chb05"
        )
    domain_train = data['domain_train']
    domain_val   = data['domain_val']
    n_domains    = int(max(domain_train.max(), domain_val.max()) + 1)
    print(f"\\n[DANN] domain_train: {len(domain_train)}  "
          f"domain_val: {len(domain_val)}  n_domains={n_domains}")"""

tb_new_2 = """# ── DANN / CORAL domain labels ────────────────────────────────────────────────
if args.dann or args.coral:
    if 'domain_train' not in data.files or 'domain_val' not in data.files:
        sys.exit(
            f"ERROR: {data_path} has no domain_train/domain_val arrays -- "
            "it predates the DANN-scoping patch to build_dataset_multi.py.\\n"
            "Rebuild via:\\n"
            "  python3 src/preprocessing/build_dataset_multi.py "
            "--patients chb01 chb02 chb05"
        )
    domain_train = data['domain_train']
    domain_val   = data['domain_val']
    n_domains    = int(max(domain_train.max(), domain_val.max()) + 1)
    print(f"\\n[domain-labels] domain_train: {len(domain_train)}  "
          f"domain_val: {len(domain_val)}  n_domains={n_domains}")"""

tb_old_3 = """    print(f"  python3 src/evaluation/eval_event_level.py --fbz "
          f"results/seizure_model_multi_dann_lambda{args.dann_lambda}"
          f"_v{args.model_version}_w4a4.fbz --eval-patient chb03  # repeat per patient")
    sys.exit(0)

if args.finetune_from:"""

tb_new_3 = """    print(f"  python3 src/evaluation/eval_event_level.py --fbz "
          f"results/seizure_model_multi_dann_lambda{args.dann_lambda}"
          f"_v{args.model_version}_w4a4.fbz --eval-patient chb03  # repeat per patient")
    sys.exit(0)

if args.coral:
    # ── CORAL training path (A-i multi-source, sec3d first screen) ────────
    from src.models.akida_cnn_v2 import (build_seizure_cnn_v2_coral,
                                          extract_deployable_submodel,
                                          make_coral_loss,
                                          CoralDistanceMonitor)
    print(f"\\n=== CORAL SCOPING EXPERIMENT (A-i multi-source, "
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

    print("\\n[CORAL] Extracting deployable submodel (trunk + main head only)...")
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
    print(f"\\n[CORAL] Covariance-distance readout, pre-training -> "
          f"post-training: {pre_d:.6f} -> {post_d:.6f}  "
          f"({'reduced' if post_d < pre_d else 'NOT reduced -- investigate '
             'before trusting this run'})")

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
    print(f"\\n[CORAL] Done. Next -- convert + eval on each held-out patient:")
    print(f"  python3 src/models/convert_to_snn.py --patient multi --base multi "
          f"--variant coral_lambda{args.coral_lambda} --eval-patient chb03 "
          f"--model-version {args.model_version} --cal-samples 1024")
    print(f"  # repeat --eval-patient for chb10 chb13 chb15 chb16 chb20")
    print(f"  python3 src/evaluation/eval_event_level.py --fbz "
          f"results/seizure_model_multi_coral_lambda{args.coral_lambda}"
          f"_v{args.model_version}_w4a4.fbz --eval-patient chb03  # repeat per patient")
    sys.exit(0)

if args.finetune_from:"""


if __name__ == '__main__':
    patch_file(AKIDA_CNN_V2_PATH, [(akida_old_1, akida_new_1)])
    patch_file(TRAIN_BASELINE_PATH, [
        (tb_old_1, tb_new_1),
        (tb_old_2, tb_new_2),
        (tb_old_3, tb_new_3),
    ])
    print("\nAll patches applied successfully.")
    print("\nSanity checks before training:")
    print("  python3 src/models/akida_cnn_v2.py   # must still print AKD1000 v1 compatible ✓")
    print("  python3 -c \"from src.models.akida_cnn_v2 import build_seizure_cnn_v2_coral; "
          "m = build_seizure_cnn_v2_coral(); m.summary()\"")
    print("  grep domain_train src/preprocessing/build_dataset_multi.py  "
          "# confirm this box's copy actually has the DANN-session patch "
          "(sec7 -- project-knowledge copy had it as of this session, "
          "but verify locally before trusting it)")
