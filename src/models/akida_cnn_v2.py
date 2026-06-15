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

import tf_keras as keras


def build_seizure_cnn_v2_3ch(n_channels: int = 18, window_samples: int = 512):
    """
    3-channel variant of seizure_cnn_v2 for Phase 2d STFT features.

    Input:  (18, 512, 3) — raw EEG + delta power envelope + beta power envelope
    Output: softmax probability over [non-seizure, seizure]

    AKD1000 v1 constraints satisfied (same as 1-channel version):
      ✓  Input channel dim = 3 (AKD1000 v1 accepts 1 or 3 — confirmed)
      ✓  No branching — strictly sequential
      ✓  Conv→Pool→ReLU block ordering
      ✓  Odd kernel heights only: (9,7) → 9 is odd, 7 is odd  ✓
      ✓  padding='valid' on first Conv2D (quantizeml 1.2.3 bug workaround)
      ✓  MaxPool padding matches preceding Conv padding in each block
      ✓  No two consecutive MaxPool layers
      ✓  No GAP→Dense  (uses Flatten instead)
      ✓  Head: Flatten → Dense(hidden) → ReLU → Dense(2, softmax)
      ✓  ReLU(max_value=6) as separate layers (not activation= inline)
      ✓  Rescaling(1/255) as first layer — input data in [0, 255]
      ✓  use_bias=False on Conv and first Dense (standard AKD1000 v1 practice)

    Parameter count:
      Slightly larger than 1-channel version due to 3× input channels in conv1:
      conv1: 32 * (9*7*3) = 6048 params  (vs 32 * (9*7*1) = 2016 in 1-ch)
      All other layers identical. Total ≈ 141,346 params.

    Usage:
      model = build_seizure_cnn_v2_3ch()
      model.summary()

      # Training: same train_baseline.py with --stft flag (loads _dataset_stft.npz)
      # Conversion: same convert_to_snn.py — no changes needed for conversion
    """
    inp = keras.Input(shape=(n_channels, window_samples, 3), name='eeg_input')

    # Rescaling layer: maps [0, 255] → [0, 1] for training stability.
    # AKD1000 v1 folds this into the input quantisation — no hardware overhead.
    x = keras.layers.Rescaling(scale=1.0 / 255.0, name='rescaling')(inp)

    # ── Block 1: spatio-temporal ──────────────────────────────────────────────
    # (9,7): 9 = spatial (channels), 7 = temporal (time samples)
    # stride (1,4) = no spatial stride, 4× temporal downsampling
    # padding='valid': REQUIRED on first Conv2D — quantizeml 1.2.3 bug
    x = keras.layers.Conv2D(
        32, (9, 7), strides=(1, 4),
        padding='valid', use_bias=False, name='conv1'
    )(x)
    x = keras.layers.MaxPooling2D((1, 2), padding='valid', name='pool1')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu1')(x)

    # ── Block 2 ───────────────────────────────────────────────────────────────
    x = keras.layers.Conv2D(
        64, (3, 3),
        padding='same', use_bias=False, name='conv2'
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu2')(x)

    # ── Block 3 ───────────────────────────────────────────────────────────────
    x = keras.layers.Conv2D(
        32, (3, 3),
        padding='same', use_bias=False, name='conv3'
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu3')(x)

    # ── Head ──────────────────────────────────────────────────────────────────
    # AKD1000 v1 valid head: Flatten → Dense(hidden) → ReLU → Dense(2, softmax)
    # GAP → Dense is NOT valid on AKD1000 v1 (confirmed constraint)
    x   = keras.layers.Flatten(name='flatten')(x)
    x   = keras.layers.Dense(64, use_bias=False, name='dense1')(x)
    x   = keras.layers.ReLU(max_value=6.0, name='relu_dense')(x)
    out = keras.layers.Dense(2, activation='softmax', name='output')(x)

    return keras.Model(inputs=inp, outputs=out, name='seizure_cnn_v2_3ch')


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import numpy as np
    model = build_seizure_cnn_v2_3ch()
    model.summary()

    # Verify forward pass
    x = np.random.rand(4, 18, 512, 3).astype('float32') * 255.0
    y = model.predict(x, verbose=0)
    print(f"\nForward pass output shape: {y.shape}  (expect (4, 2))")
    print(f"Softmax row sums: {y.sum(axis=1)}  (expect all ≈ 1.0)")
    assert y.shape == (4, 2), f"Wrong output shape: {y.shape}"
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5), "Softmax not normalised"
    print("\n3-channel model smoke test PASSED ✓")
    print(f"Total params: {model.count_params():,}")


def build_seizure_cnn_v2(n_channels: int = 18, window_samples: int = 512):
    """
    Standard 1-channel v2 architecture for Phase 2a/2c/2e.

    Input:  (18, 512, 1)
    Output: softmax probability over [non-seizure, seizure]

    Restored in Phase 2e after Phase 2d STFT refactor removed it.
    Architecture identical to 3ch variant except input channels=1.
    Total params: ~137,000
    """
    inp = keras.Input(shape=(n_channels, window_samples, 1), name='eeg_input')

    x = keras.layers.Rescaling(scale=1.0 / 255.0, name='rescaling')(inp)

    # Block 1: spatio-temporal
    x = keras.layers.Conv2D(
        32, (9, 7), strides=(1, 4),
        padding='valid', use_bias=False, name='conv1'
    )(x)
    x = keras.layers.MaxPooling2D((1, 2), padding='valid', name='pool1')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu1')(x)

    # Block 2
    x = keras.layers.Conv2D(
        64, (3, 3),
        padding='same', use_bias=False, name='conv2'
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu2')(x)

    # Block 3
    x = keras.layers.Conv2D(
        32, (3, 3),
        padding='same', use_bias=False, name='conv3'
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu3')(x)

    # Head
    x   = keras.layers.Flatten(name='flatten')(x)
    x   = keras.layers.Dense(64, use_bias=False, name='dense1')(x)
    x   = keras.layers.ReLU(max_value=6.0, name='relu_dense')(x)
    out = keras.layers.Dense(2, activation='softmax', name='output')(x)

    return keras.Model(inputs=inp, outputs=out, name='seizure_cnn_v2')


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
