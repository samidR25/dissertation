"""
Full pipeline smoke test — May 2026 stack.

Covers: TF 2.19 GPU → QuantizeML quantisation (per-tensor, AKD1000 v1)
        → cnn2snn AkidaVersion.v1 conversion → akida simulator → .fbz save

Key checks vs older tutorials:
  - import tf_keras as keras (NOT from tensorflow import keras)
  - QuantizationParams(per_tensor_activations=True) for AKD1000 v1
  - set_akida_version(AkidaVersion.v1) context manager
  - Saved model is .fbz not .h5

Source: BrainChip MetaTF docs https://doc.brainchipinc.com
"""
import numpy as np
import tensorflow as tf
import tf_keras as keras
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import (convert, check_model_compatibility,
                     set_akida_version, AkidaVersion)
import akida, os

os.makedirs('results', exist_ok=True)

# ── 0. GPU check ──────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus}")
assert gpus, "No GPU — check CUDA 12.4"
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── 1. MNIST ──────────────────────────────────────────────────────
(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
x_tr = x_tr[..., np.newaxis].astype('float32') / 255.0
x_te = x_te[..., np.newaxis].astype('float32') / 255.0

# ── 2. Build ANN ──────────────────────────────────────────────────
# Rules: ReLU only, 3x3/1x1 kernels, no BatchNorm before activation
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='valid', input_shape=(28,28,1)),
    keras.layers.ReLU(max_value=6),
    keras.layers.MaxPooling2D(2, padding='valid'),
    keras.layers.Conv2D(64, 3, padding='valid'),
    keras.layers.ReLU(max_value=6),
    keras.layers.MaxPooling2D(2, padding='valid'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

# ── 3. Quantize first, then check AKD1000 v1 compatibility ───────
from quantizeml.models import quantize, QuantizationParams
qparams = QuantizationParams(weight_bits=4, activation_bits=4,
                              per_tensor_activations=True)
model_q = quantize(model, qparams=qparams,
                   samples=x_tr[:100])

with set_akida_version(AkidaVersion.v1):
    check_model_compatibility(model)  # raises if incompatible
print("AKD1000 v1 compatible ✓")
# ── 4. Train on GPU ───────────────────────────────────────────────
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_tr, y_tr, epochs=3, batch_size=128,
          validation_split=0.1, verbose=1)
_, ann_acc = model.evaluate(x_te, y_te, verbose=0)
print(f"ANN accuracy: {ann_acc:.4f}")
model.save('results/hello_ann.h5')

# ── 5. QuantizeML — AKD1000 v1 params ────────────────────────────
# per_tensor_activations=True is MANDATORY for AKD1000 v1 hardware.
# The default (per_tensor_activations=False) produces per-axis
# quantisation that the AKD1000 chip cannot execute.
# Source: BrainChip QuantizeML docs, AKD1000 v1 constraints section.
qparams = QuantizationParams(
    input_weight_bits=8,
    weight_bits=4,
    activation_bits=4,
    per_tensor_activations=True   # MANDATORY for AKD1000 v1
)
q_model = quantize(model, qparams=qparams,
                   samples=x_te[:256])   # calibration samples

# Fine-tune to recover accuracy lost in quantisation
q_model.compile(optimizer=keras.optimizers.Adam(1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
q_model.fit(x_tr, y_tr, epochs=2, batch_size=128,
            validation_split=0.1, verbose=1)
_, q_acc = q_model.evaluate(x_te, y_te, verbose=0)
print(f"Quantised accuracy: {q_acc:.4f}")

# ── 6. Convert to SNN — must target AKD1000 v1 ───────────────────
# Default convert() targets Akida 2.0 which the AKD1000 cannot run.
# Use set_akida_version(AkidaVersion.v1) context manager always.
with set_akida_version(AkidaVersion.v1):
    akida_model = convert(q_model,
                          file_path='results/hello_akida.fbz')
# Saved as .fbz — new format for Akida models (replaces .h5)
print("Converted → results/hello_akida.fbz")

# ── 7. Simulator evaluation ───────────────────────────────────────
loaded = akida.Model('results/hello_akida.fbz')
preds  = loaded.predict(x_te[:500])
snn_acc = np.mean(np.argmax(preds.squeeze(), axis=1) == y_te[:500])
drop    = ann_acc - snn_acc
print(f"SNN simulator accuracy: {snn_acc:.4f}")
print(f"Accuracy drop: {drop:.4f} "
      f"({'✓ acceptable' if drop < 0.05 else '✗ investigate'})")

stats = loaded.statistics
print("\nSpike activity per layer (lower = sparser = less power):")
print(stats)
print("\nSmoke test: PASSED ✓" if drop < 0.05 else "Smoke test: FAILED — check quantisation")
