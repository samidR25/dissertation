"""
AKD1000 v1 compatible CNN for CHB-MIT seizure detection.
Confirmed compatible: 31 May 2026 (akida 2.19.1 / cnn2snn 2.19.1)

Input:  (N, 18, 512, 1)  —  18 EEG channels × 512 samples (2s @ 256Hz) × 1
Output: (N, 2)           —  [P(non-seizure), P(seizure)]

Architecture constraints — ALL required for AKD1000 v1:
   import tf_keras as keras  (never tensorflow.keras)
   padding='valid' on first Conv2D  (quantizeml 1.2.3 strips 'same' on input layer)
   keras.layers.ReLU(max_value=6) as SEPARATE layer  (not activation='relu' inline)
   Block order: Conv → Pool → ReLU  (pool before activation — confirmed working)
   MaxPool padding matches Conv padding within same block
   No GAP before Dense  (GAP→Dense fails AKD1000 v1 — probe3 G)
   Head: Flatten → Dense → ReLU → Dense(softmax)  (probe3 F confirmed)
   Dropout between blocks only  (after ReLU, before next Conv — probe3 E confirmed)
   Dense(2, softmax) output  (not Dense(1, sigmoid) — MetaTF requires multi-class)
   check_model_compatibility() on FLOAT model only — returns None, use try/except

Run:
    python3 code/models/akida_cnn.py
"""
import sys
import tensorflow as tf
import tf_keras as keras
from cnn2snn import check_model_compatibility, set_akida_version, AkidaVersion


def build_seizure_cnn(n_channels: int = 18,
                      window_samples: int = 512) -> keras.Model:
    """
    Build AKD1000 v1 compatible CNN for seizure detection.

    Architecture (3 conv blocks + Flatten head):
      Block 1: (1,7) kernel, stride (1,4) — temporal patterns within channels
               Aggressive time-dim reduction: 512 → 127 → 63 samples
      Block 2: (3,3) kernel — cross-channel + temporal feature integration
      Block 3: (3,3) kernel — higher-level ictal feature detection
      Head:    Flatten → Dense(64) → ReLU → Dense(2, softmax)

    Design rationale:
      - Shallow (3 blocks): prevents overfit on 8,980 training windows;
        fits in AKD1000 v1 on-chip SRAM (201k params → 785 KB)
      - (1,7) block 1 kernel: operates within each electrode independently
        first — biologically motivated (local field potential patterns)
      - Flatten head over GAP: GAP→Dense fails AKD1000 v1 (probe3 G);
        Flatten→Dense head is confirmed valid (probe3 F, M)
      - No Dropout inside blocks: only safe between blocks (probe3 E)

    Args:
        n_channels:     EEG channels — 18 for CHB-MIT common subset
        window_samples: Time samples — 512 for 2s @ 256Hz; 256 for 1s ablation

    Returns:
        Uncompiled float Keras model, AKD1000 v1 compatible.
    """
    inp = keras.Input(shape=(n_channels, window_samples, 1),
                      name='eeg_input')
        # Rescaling: [0,255] inputs → [0,1] for the network internals.
    # Must be first layer — AKD1000 v1 constraint (compatibility_checks.py).
    # Means training, conversion, and hardware inference all use identical
    # [0,255] inputs with no per-script scaling logic anywhere.
    x = keras.layers.Rescaling(scale=1.0/255.0, name='rescaling')(inp)

    # ── Block 1 — temporal (per-channel) ─────────────────────────────────────
    # padding='valid' on FIRST Conv2D — quantizeml 1.2.3 bug strips 'same' here.
    # (1,7) kernel: spans 7 time steps, 1 channel → per-electrode temporal filter.
    # stride (1,4): reduces time dim 512 → 127, cutting compute significantly.
    x = keras.layers.Conv2D(32, (1, 7), strides=(1, 4), padding='valid',
                            use_bias=False, name='conv1')(x)
    # MaxPool before ReLU: confirmed working order (probe3 A).
    # padding='valid' must match conv1's padding (constraint #12).
    x = keras.layers.MaxPooling2D((1, 2), padding='valid', name='pool1')(x)
    x = keras.layers.ReLU(max_value=6, name='relu1')(x)
    # output: (N, 18, 63, 32)

    # ── Block 2 — spatial+temporal ───────────────────────────────────────────
    # (3,3): integrates across channels and time simultaneously.
    # padding='same': preserves spatial dims → pool halves them cleanly.
    x = keras.layers.Conv2D(64, (3, 3), padding='same',
                            use_bias=False, name='conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = keras.layers.ReLU(max_value=6, name='relu2')(x)
    # output: (N, 9, 32, 64)

    # ── Block 3 — higher-level ictal features ────────────────────────────────
    x = keras.layers.Conv2D(32, (3, 3), padding='same',
                            use_bias=False, name='conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(x)
    x = keras.layers.ReLU(max_value=6, name='relu3')(x)
    # output: (N, 5, 16, 32)

    # Dropout here (between blocks, after relu3) is valid if needed for overfit.
    # Uncomment if Gate 3 shows train/val gap > 0.30:
    # x = keras.layers.Dropout(0.3, name='dropout_pre_head')(x)

    # ── Head ─────────────────────────────────────────────────────────────────
    # Flatten → Dense → ReLU → Dense(softmax) — confirmed valid (probe3 F, M).
    # GAP → Dense is NOT valid in AKD1000 v1 (probe3 G confirmed failure).
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(64, use_bias=False, name='dense1')(x)
    x = keras.layers.ReLU(max_value=6, name='relu_dense')(x)
    out = keras.layers.Dense(2, activation='softmax', name='output')(x)
    # softmax head is NOT quantised — MetaTF appends a dequantizer here.
    # "Conversion stops at layer output because of a dequantizer" is EXPECTED.

    return keras.Model(inputs=inp, outputs=out, name='seizure_cnn')


if __name__ == '__main__':
    model = build_seizure_cnn(n_channels=18, window_samples=512)
    model.summary()

    total = model.count_params()
    print(f"\nTotal parameters: {total:,}")
    if total > 500_000:
        print("⚠  > 500k params — may exceed AKD1000 v1 on-chip SRAM.")
        print("   Reduce conv filters or window_samples.")

    # ── AKD1000 v1 compatibility check ───────────────────────────────────────
    # check_model_compatibility() in cnn2snn 2.19.1 returns None — NOT (ok, issues).
    # Must be called on FLOAT model only — not on a quantised model.
    # Must wrap in set_akida_version(AkidaVersion.v1).
    print("\n=== AKD1000 v1 compatibility check ===")
    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(model)
        print("AKD1000 v1 compatible — safe to proceed to smoke_test.py")
    except Exception as e:
        print(f"INCOMPATIBLE   {e}")
        print("Fix architecture above before proceeding.")
        sys.exit(1)
