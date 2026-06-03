"""
AKD1000 v1 compatible CNN v2 — Spatio-Temporal Architecture
for CHB-MIT seizure detection.

Architecture rationale
----------------------
v1 used a (1,7) kernel in block 1, treating all 18 EEG channels
as completely independent. This misses the defining characteristic
of seizure activity: consistent spatial propagation patterns across
electrode groups (Litt & Echauz, 2002; Tsiouris et al., 2018).

v2 replaces block 1 with a (9,7) spatio-temporal kernel, spanning
9 of 18 electrodes simultaneously. This covers approximately half
the electrode array — consistent with the spatial extent of typical
ictal discharges in CHB-MIT recordings (Shoeb & Guttag, 2010).

The (9,7) kernel was selected empirically via probe testing against
AKD1000 v1 hardware constraints:
  - (17,7): passes but collapses spatial dim to 2 (too aggressive)
  - (9,7):  passes, preserves 10 spatial positions for later blocks
  - (3,7):  passes but too local to capture ictal propagation extent

AKD1000 v1 kernel constraints discovered via probing (June 2026):
  - Both kernel dimensions must be odd OR height=1
  - Even kernel heights (2, 4, 6, 18...) fail at convert stage
  - Kernel height must be < input height (19+ fails with negative dim)
  - Branching (Concatenate) not supported in v1
  - DepthwiseConv2D / SeparableConv2D cannot be first layer

Patient-specific fine-tuning
-----------------------------
After base training, the model can be fine-tuned per patient by
freezing all layers except dense1 and relu_dense, then training
for 20 epochs on patient-specific data. This adapts the
classification head to patient-specific ictal signatures while
preserving the general spatio-temporal feature extraction.

References
----------
Litt & Echauz (2002). Prediction of seizures. Lancet Neurology 1(1), 22-30.
Shoeb & Guttag (2010). Application of ML to epileptic seizure detection.
  ICML 2010.
Tsiouris et al. (2018). A long short-term memory deep learning network
  for the prediction of epileptic seizures. IEEE TNSRE 26(10), 1944-1956.

Input shape:  (N, 18, 512, 1)  — [0, 255] float32 (pre-scaled by build_dataset.py)
Output shape: (N, 2)           — [P(non-seizure), P(seizure)]

Run:
    python3 src/models/akida_cnn_v2.py
"""
import sys
import tensorflow as tf
import tf_keras as keras
from cnn2snn import check_model_compatibility, set_akida_version, AkidaVersion


def build_seizure_cnn_v2(n_channels: int = 18,
                          window_samples: int = 512) -> keras.Model:
    """
    Build AKD1000 v1 compatible CNN v2 for seizure detection.

    Key differences from v1:
      - Block 1 uses (9,7) spatio-temporal kernel instead of (1,7) temporal-only
      - Spatial receptive field covers 9/18 electrodes simultaneously
      - Learns electrode co-activation patterns jointly with temporal dynamics
      - Slightly fewer parameters (~140k vs 201k) — less overfit risk

    Args:
        n_channels:     EEG channels — 18 for CHB-MIT common subset
        window_samples: Time samples — 512 for 2s @ 256Hz

    Returns:
        Uncompiled float Keras model, AKD1000 v1 compatible.
    """
    inp = keras.Input(shape=(n_channels, window_samples, 1),
                      name='eeg_input')

    # Rescaling: [0,255] → [0,1] for network internals.
    # AKD1000 InputConvolutional expects [0,255] at input;
    # Rescaling layer is the first layer and handles the division.
    x = keras.layers.Rescaling(scale=1.0/255.0, name='rescaling')(inp)

    # ── Block 1 — spatio-temporal ─────────────────────────────────────────────
    # (9,7) kernel: spans 9 channels × 7 timesteps simultaneously.
    # Learns spatial co-activation patterns (which electrode groups
    # activate together) jointly with local temporal oscillatory patterns.
    # stride=(1,4): reduces time dim 512→127, aggressive temporal downsampling.
    # padding='valid': required on first Conv2D (quantizeml 1.2.3 constraint).
    # Output: (None, 10, 127, 32) → after pool: (None, 10, 63, 32)
    x = keras.layers.Conv2D(32, (9, 7), strides=(1, 4), padding='valid',
                            use_bias=False, name='conv1')(x)
    x = keras.layers.MaxPooling2D((1, 2), padding='valid', name='pool1')(x)
    x = keras.layers.ReLU(max_value=6, name='relu1')(x)

    # ── Block 2 — cross-regional integration ─────────────────────────────────
    # (3,3): integrates across the 10 remaining spatial positions and
    # refines temporal features. Learns which spatial patterns (from block 1)
    # co-occur across different electrode regions.
    # Output after pool: (None, 5, 32, 64)
    x = keras.layers.Conv2D(64, (3, 3), padding='same',
                            use_bias=False, name='conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = keras.layers.ReLU(max_value=6, name='relu2')(x)

    # ── Block 3 — high-level ictal features ──────────────────────────────────
    # (3,3): learns high-level combinations of cross-regional patterns.
    # At this depth the receptive field covers the full recording segment.
    # Output after pool: (None, 3, 16, 32)
    x = keras.layers.Conv2D(32, (3, 3), padding='same',
                            use_bias=False, name='conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(x)
    x = keras.layers.ReLU(max_value=6, name='relu3')(x)

    # Dropout can be added here between blocks if Gate 3 shows overfit > 0.30
    # x = keras.layers.Dropout(0.3, name='dropout_pre_head')(x)

    # ── Head ─────────────────────────────────────────────────────────────────
    # Flatten → Dense → ReLU → Dense(softmax) — confirmed valid (probe3 F, M).
    # Dense(64) is the patient-specific fine-tuning target: freeze all other
    # layers and fine-tune dense1 + relu_dense for per-patient adaptation.
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(64, use_bias=False, name='dense1')(x)
    x = keras.layers.ReLU(max_value=6, name='relu_dense')(x)
    out = keras.layers.Dense(2, activation='softmax', name='output')(x)

    return keras.Model(inputs=inp, outputs=out, name='seizure_cnn_v2')


def build_patient_adapted_model(base_model: keras.Model,
                                 freeze_until: str = 'relu3') -> keras.Model:
    """
    Prepare a copy of the base model for patient-specific fine-tuning.

    Freezes all layers up to and including freeze_until, leaving only
    the Dense head trainable. Call this after loading a trained base model,
    then compile and fit for 20 epochs on patient-specific data.

    Args:
        base_model:    Trained base model from build_seizure_cnn_v2()
        freeze_until:  Name of last frozen layer (default: 'relu3')

    Returns:
        Model with frozen feature extractor, trainable Dense head.
    """
    # Rebuild from same config to get a fresh copy
    adapted = build_seizure_cnn_v2()
    adapted.set_weights(base_model.get_weights())

    freeze = True
    for layer in adapted.layers:
        if freeze:
            layer.trainable = False
        if layer.name == freeze_until:
            freeze = False  # everything after this is trainable

    trainable = [l.name for l in adapted.layers if l.trainable]
    frozen    = [l.name for l in adapted.layers if not l.trainable]
    print(f"Frozen layers  : {frozen}")
    print(f"Trainable layers: {trainable}")
    return adapted


if __name__ == '__main__':
    model = build_seizure_cnn_v2(n_channels=18, window_samples=512)
    model.summary()

    total = model.count_params()
    print(f"\nTotal parameters: {total:,}")

    # Compare to v1
    print(f"v1 parameters:  201,058")
    print(f"v2 parameters:  {total:,}  ({total - 201058:+,} vs v1)")

    if total > 500_000:
        print("⚠  > 500k params — may exceed AKD1000 v1 on-chip SRAM.")

    # ── AKD1000 v1 compatibility check ───────────────────────────────────────
    print("\n=== AKD1000 v1 compatibility check ===")
    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(model)
        print("AKD1000 v1 compatible ✓ — safe to train")
    except Exception as e:
        print(f"INCOMPATIBLE ✗  {e}")
        sys.exit(1)

    # ── Patient adaptation demo ───────────────────────────────────────────────
    print("\n=== Patient adaptation layer freeze test ===")
    adapted = build_patient_adapted_model(model)
    adapted.summary()
