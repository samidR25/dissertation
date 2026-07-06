"""
PATCH FOR: src/models/akida_cnn_v2.py
======================================
Add this function to akida_cnn_v2.py (anywhere after build_seizure_cnn_v2).

This is the 3-channel variant for Phase 2d STFT features.
All AKD1000 v1 constraints are preserved — no new constraints introduced.
AKD1000 v1 explicitly accepts input channel dim = 1 or 3 (confirmed constraint).

Apply with:
  cat >> src/models/akida_cnn_v2.py << 'PATCH'
  [paste function below]
  PATCH

Or simply open akida_cnn_v2.py in hx and append at the end.
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
import tf_keras as keras
import tensorflow as tf


def build_seizure_cnn_v2_3ch(n_channels: int = 18, window_samples: int = 512):
    """
    3-channel variant of seizure_cnn_v2. Originally built for Phase 2d STFT
    features; reused for Gate 2's long-context channels (Arms B/C) per
    Handoff_architecture_scoping_to_implementation.md §10.

    Gate 2 BN fix: BatchNormalization added after each Conv2D, before the
    corresponding Pool — Conv → BN → Pool → ReLU(6) — mirroring EXACTLY the
    fix already applied to the 1-channel build_seizure_cnn_v2 (Phase 3
    overhaul). This was NOT in the original 3ch variant — confirmed by
    direct inspection, it predates that fix and was explicitly left
    untouched at the time ("out of scope, not part of the active
    pipeline"). Without it, the freeze-depth fine-tune path's BN-re-
    estimation adaptation mechanism (Gate 2b — the actual mechanism behind
    chb10ft's result) has nothing to act on for this architecture, breaking
    Gate 2's "same fine-tune path" comparability requirement at the
    mechanism level, not just the flag-name level.

    Input:  (18, window_samples, 3) — raw EEG + 2 derived channels
    Output: softmax probability over [non-seizure, seizure]

    AKD1000 v1 constraints satisfied (re-verified end-to-end — compat
    check -> quantize -> convert — after this BN addition, both window
    sizes: 141,858 params @ ws=512, 191,010 @ ws=768):
      ✓  Input channel dim = 3 (AKD1000 v1 accepts 1 or 3 — confirmed)
      ✓  No branching — strictly sequential
      ✓  Conv→BN→Pool→ReLU block ordering (confirmed-compatible per
         probe3c/d, same ordering already validated for the 1ch BN fix)
      ✓  Odd kernel heights only: (9,7) → 9 is odd, 7 is odd  ✓
      ✓  padding='valid' on first Conv2D (quantizeml 1.2.3 bug workaround)
      ✓  MaxPool padding matches preceding Conv padding in each block
      ✓  No two consecutive MaxPool layers
      ✓  No GAP→Dense  (uses Flatten instead)
      ✓  Head: Flatten → Dense(hidden) → ReLU → Dense(2, softmax)
      ✓  ReLU(max_value=6) as separate layers (not activation= inline)
      ✓  Rescaling(1/255) as first layer — input data in [0, 255]
      ✓  use_bias=False on Conv and first Dense (standard AKD1000 v1 practice)

    BN folds into the preceding Conv's weights at convert() time, so it
    costs nothing at inference — purely a training-stability/adaptation
    intervention, same rationale as the 1-channel fix.
    """
    inp = keras.Input(shape=(n_channels, window_samples, 3), name='eeg_input')

    x = keras.layers.Rescaling(scale=1.0 / 255.0, name='rescaling')(inp)

    # ── Block 1: spatio-temporal ──────────────────────────────────────────────
    x = keras.layers.Conv2D(
        32, (9, 7), strides=(1, 4),
        padding='valid', use_bias=False, name='conv1'
    )(x)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.MaxPooling2D((1, 2), padding='valid', name='pool1')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu1')(x)

    # ── Block 2 ───────────────────────────────────────────────────────────────
    x = keras.layers.Conv2D(
        64, (3, 3),
        padding='same', use_bias=False, name='conv2'
    )(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu2')(x)

    # ── Block 3 ───────────────────────────────────────────────────────────────
    x = keras.layers.Conv2D(
        32, (3, 3),
        padding='same', use_bias=False, name='conv3'
    )(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu3')(x)

    # ── Head ──────────────────────────────────────────────────────────────────
    x   = keras.layers.Flatten(name='flatten')(x)
    x   = keras.layers.Dense(64, use_bias=False, name='dense1')(x)
    x   = keras.layers.ReLU(max_value=6.0, name='relu_dense')(x)
    out = keras.layers.Dense(2, activation='softmax', name='output')(x)

    return keras.Model(inputs=inp, outputs=out, name='seizure_cnn_v2_3ch')


def build_seizure_cnn_v2(n_channels: int = 18, window_samples: int = 512):
    """
    Standard 1-channel v2 architecture for Phase 2a/2c/2e.

    Phase 3 update: BatchNormalization added after each Conv2D, before the
    corresponding Pool — Conv → BN → Pool → ReLU(6) — confirmed-compatible
    ordering per probe3c/d. BN folds into the preceding Conv's weights at
    convert() time, so it costs nothing at inference; this is purely a
    training-stability intervention aimed at the SNN-conversion collapse
    diagnosed in Phase 2e (see Handoff_phase3_overhaul.md §1).
    """
    inp = keras.Input(shape=(n_channels, window_samples, 1), name='eeg_input')
    x = keras.layers.Rescaling(scale=1.0 / 255.0, name='rescaling')(inp)

    # Block 1: spatio-temporal
    x = keras.layers.Conv2D(32, (9, 7), strides=(1, 4),
        padding='valid', use_bias=False, name='conv1')(x)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.MaxPooling2D((1, 2), padding='valid', name='pool1')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu1')(x)

    # Block 2
    x = keras.layers.Conv2D(64, (3, 3),
        padding='same', use_bias=False, name='conv2')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu2')(x)

    # Block 3
    x = keras.layers.Conv2D(32, (3, 3),
        padding='same', use_bias=False, name='conv3')(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu3')(x)

    # Head
    x   = keras.layers.Flatten(name='flatten')(x)
    x   = keras.layers.Dense(64, use_bias=False, name='dense1')(x)
    x   = keras.layers.ReLU(max_value=6.0, name='relu_dense')(x)
    out = keras.layers.Dense(2, activation='softmax', name='output')(x)

    return keras.Model(inputs=inp, outputs=out, name='seizure_cnn_v2')

class GradientReversalLayer(keras.layers.Layer):
    """
    Gradient Reversal Layer (Ganin & Lempitsky, 2015) for DANN.
    Forward: identity. Backward: gradient scaled by -lambda_.

    TRAINING-ONLY -- never part of the deployed graph. Discarded (along
    with the whole domain head) before quantize()/convert() via
    extract_deployable_submodel() below. AKD1000 v1 never sees this
    layer or the branching graph it creates; the deployed model stays
    architecturally identical to build_seizure_cnn_v2 (Handoff_
    calibration_session_to_dann_supcon.md sec4b; confirmed against
    Akd1000_v1_architecture_constraints.md, which forbids branching
    only in the CONVERTED graph, not the training-time TF graph).
    """
    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self._lambda = float(lambda_)

    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                return -self._lambda * dy
            return x, grad
        return _reverse(x)

    def get_config(self):
        config = super().get_config()
        config.update({'lambda_': self._lambda})
        return config


def build_seizure_cnn_v2_dann(n_channels=18, window_samples=512,
                               n_domains=3, grl_lambda=1.0):
    """
    DANN scoping experiment (Handoff_calibration_session_to_dann_supcon.md
    sec4d/4e), first concrete run. Domains = pool patients, in the order
    passed to build_dataset_multi.py's --patients (domain id = list
    index). For the first screen: domains = chb01/02/05 -> n_domains=3.

    Shared trunk uses IDENTICAL layer names to build_seizure_cnn_v2, so
    extract_deployable_submodel() below can copy weights by name with no
    remapping. Domain head attaches to the flattened shared features via
    a GradientReversalLayer -- training-only, discarded before deployment.
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

    # -- Domain classifier head (training-only, via GRL) ------------------
    d = GradientReversalLayer(lambda_=grl_lambda, name='grl')(shared)
    d = keras.layers.Dense(32, use_bias=False, name='domain_dense1')(d)
    d = keras.layers.ReLU(max_value=6.0, name='domain_relu1')(d)
    domain_out = keras.layers.Dense(n_domains, activation='softmax',
                                     name='domain_output')(d)

    return keras.Model(inputs=inp, outputs=[main_out, domain_out],
                        name='seizure_cnn_v2_dann')


def extract_deployable_submodel(dann_model, n_channels=18, window_samples=512):
    """
    Strip a trained DANN model down to plain build_seizure_cnn_v2 --
    copies weights by matching layer name for the shared trunk + main
    head, drops the GRL and domain head entirely. This is the model
    that goes through quantize()/convert(); the DANN model itself
    never does (Handoff sec4b).
    """
    deployable = build_seizure_cnn_v2(n_channels=n_channels,
                                       window_samples=window_samples)
    copied, skipped = [], []
    for layer in deployable.layers:
        try:
            src_layer = dann_model.get_layer(layer.name)
        except ValueError:
            skipped.append(layer.name)
            continue
        layer.set_weights(src_layer.get_weights())
        copied.append(layer.name)
    print(f"[extract_deployable_submodel] Copied: {copied}")
    if skipped:
        print(f"[extract_deployable_submodel] WARNING -- no source layer "
              f"for: {skipped} (expected empty; investigate before trusting "
              "this checkpoint)")
    return deployable


def build_seizure_cnn_v2_coral(n_channels=18, window_samples=512):
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
        print(f"\n[CORAL] Pre-training val covariance-distance: {d0:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        d = self._measure()
        self.history.append((f'epoch_{epoch}', d))
        print(f"[CORAL] Epoch {epoch} val covariance-distance: {d:.6f}")


def build_seizure_cnn_v2_ssl_pretrain(n_channels=18, window_samples=512, mask_len=128):
    """
    Candidate C-i pretext-task model (Handoff_post_dann_scoping_to_
    implementation.md sec5c). Trunk (rescaling -> flatten) is IDENTICAL
    (same layer names) to build_seizure_cnn_v2 -- extract_pretrained_
    trunk() below copies these weights with zero remapping. Decoder head
    is a plain 2-layer MLP reconstructing the masked time-span (n_channels
    x mask_len, flattened then reshaped) from the 1536-dim bottleneck --
    training-only, discarded entirely before the supervised phase.

    Input is the MASKED window (masked span already zeroed by the caller,
    see pretrain_ssl.py) -- this model only ever sees masked input, never
    the clean window, by construction.
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

    # -- Decoder head (training-only, discarded before the supervised
    # phase -- reconstructs the masked [n_channels, mask_len] patch) --
    d = keras.layers.Dense(256, activation='relu', name='ssl_decoder_dense1')(shared)
    d = keras.layers.Dense(n_channels * mask_len, activation='linear',
                            name='ssl_decoder_out')(d)
    recon = keras.layers.Reshape((n_channels, mask_len), name='ssl_reconstruction')(d)

    return keras.Model(inputs=inp, outputs=recon, name='seizure_cnn_v2_ssl_pretrain')


def extract_pretrained_trunk(ssl_model, n_channels=18, window_samples=512):
    """
    Copy ONLY the trunk layers (rescaling through flatten) by name from a
    trained C-i pretext model into a plain build_seizure_cnn_v2 instance.
    The Dense head (dense1/relu_dense/output) is deliberately left at
    ITS OWN fresh random init -- C-i's pretext task pretrains the
    representation extractor, not the classifier; the supervised phase
    trains the head from scratch on top of the pretrained trunk.
    """
    target = build_seizure_cnn_v2(n_channels=n_channels, window_samples=window_samples)
    TRUNK_LAYERS = {'rescaling', 'conv1', 'bn1', 'pool1', 'relu1',
                     'conv2', 'bn2', 'pool2', 'relu2',
                     'conv3', 'bn3', 'pool3', 'relu3', 'flatten'}
    copied, skipped_head = [], []
    for layer in target.layers:
        if layer.name not in TRUNK_LAYERS:
            skipped_head.append(layer.name)   # expected -- head stays random
            continue
        try:
            src_layer = ssl_model.get_layer(layer.name)
        except ValueError:
            print(f"[extract_pretrained_trunk] WARNING -- no source layer "
                  f"for trunk layer '{layer.name}' -- investigate before "
                  "trusting this checkpoint.")
            continue
        layer.set_weights(src_layer.get_weights())
        copied.append(layer.name)
    print(f"[extract_pretrained_trunk] Copied (pretrained): {copied}")
    print(f"[extract_pretrained_trunk] Left at random init (head, by "
          f"design): {skipped_head}")
    return target


def build_patient_adapted_model(base_model, freeze_until='relu3'):
    """
    Clone base_model and freeze layers up to and including freeze_until.
    Used for patient-specific fine-tuning from a pre-trained base.

    Args:
        base_model   : trained keras.Model (e.g. best_ann_chb01_v2.h5)
        freeze_until : name of last frozen layer (default 'relu3' —
                       freezes conv extractor, trains Dense head only)

    Returns:
        adapted model ready for fine-tuning with compile() + fit()

    Frozen / trainable parameter counts (freeze_until='relu3'):
        Frozen    : conv1+pool1+relu1+conv2+pool2+relu2+conv3+pool3+relu3
                    = ~38,880 params
        Trainable : flatten+dense1+relu_dense+output
                    = ~98,434 params

    Usage:
        base = keras.models.load_model('results/best_ann_chb01_v2.h5')
        adapted = build_patient_adapted_model(base, freeze_until='relu3')
        adapted.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        adapted.fit(X_train, y_train, epochs=20, ...)
    """
    # Freeze all layers up to and including freeze_until
    freeze = True
    for layer in base_model.layers:
        if freeze:
            layer.trainable = False
        if layer.name == freeze_until:
            freeze = False  # unfreeze everything after this layer

    frozen    = sum(1 for l in base_model.layers if not l.trainable)
    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"Frozen layers    : {frozen}")
    print(f"Trainable layers : {trainable}")
    print(f"Frozen params    : {sum(l.count_params() for l in base_model.layers if not l.trainable):,}")
    print(f"Trainable params : {sum(l.count_params() for l in base_model.layers if l.trainable):,}")

    return base_model
if __name__ == '__main__':
    import numpy as np
    from cnn2snn import check_model_compatibility, set_akida_version, AkidaVersion

    # ── 3-channel variant (Phase 2d STFT) — forward pass sanity only ─────────
    model_3ch = build_seizure_cnn_v2_3ch()
    x3 = np.random.rand(4, 18, 512, 3).astype('float32') * 255.0
    y3 = model_3ch.predict(x3, verbose=0)
    assert y3.shape == (4, 2), f"Wrong output shape: {y3.shape}"
    assert np.allclose(y3.sum(axis=1), 1.0, atol=1e-5), "Softmax not normalised"
    print(f"3-channel model: forward pass OK — {model_3ch.count_params():,} params")

    # ── 1-channel variant (active Phase 2e/3 architecture) ───────────────────
    model = build_seizure_cnn_v2()
    model.summary()

    print("\nChecking AKD1000 v1 compatibility (1-channel)...")
    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(model)
        print("AKD1000 v1 compatible ✓")
    except Exception as e:
        print(f"INCOMPATIBLE: {e}")
        raise

    x = np.random.rand(4, 18, 512, 1).astype('float32') * 255.0
    y = model.predict(x, verbose=0)
    assert y.shape == (4, 2), f"Wrong output shape: {y.shape}"
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5), "Softmax not normalised"
    print(f"1-channel model: forward pass OK — {model.count_params():,} params")
