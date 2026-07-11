# AKD1000 v1 — Definitive Architecture Constraints

**Last updated: 3 June 2026 | Empirically verified via probe1–probe3b + v2 architecture probing**

Stack: `akida 2.19.1 / cnn2snn 2.19.1 / quantizeml 1.2.3 / tf-keras 2.19 / TF 2.19.0`

Each pattern tested through all three stages: `check_model_compatibility()` on float model → `quantize()` on float model → `convert()` on quantized model.

---

## 1. Empirical probe results

### probe3 — block pattern tests (13 patterns)

|Lbl|Pattern|Result|Stage failed|
|---|---|---|---|
|A|`Conv → Pool → ReLU`|✅ PASS|—|
|B|`Conv → ReLU → Pool`|✅ PASS|—|
|C|`Conv → BN → Pool → ReLU`|✅ PASS|—|
|D|`Conv → BN → ReLU` (no pool)|✅ PASS|—|
|E|`Dropout(0.3)` between blocks|✅ PASS|—|
|F|`Flatten → Dense → ReLU → Dense(softmax)` head|✅ PASS|—|
|G|`Conv → GAP → Dense`|❌ FAIL|compat|
|H|`Conv → GAP → ReLU` (then Flatten head)|✅ PASS|—|
|I|`Conv → ReLU` stacked, no pool|✅ PASS|—|
|J|`SeparableConv → Pool → ReLU`|✅ PASS|—|
|K|`Conv(activation='relu')` inline|❌ FAIL|compat|
|L|Two consecutive MaxPool|❌ FAIL|compat|
|M|Full 3-block EEG architecture (akida_cnn.py)|✅ PASS|—|

### v2 architecture probes — kernel dimension constraints (June 2026)

|Kernel|Result|Notes|
|---|---|---|
|(1,7)|✅ PASS|height=1 valid (special case) — used in v1 architecture|
|(3,7)|✅ PASS|odd height|
|(5,7)|✅ PASS|odd height|
|(7,7)|✅ PASS|odd height|
|(9,7)|✅ PASS|odd height — **used in v2 architecture**|
|(17,7)|✅ PASS|odd height, 18-17=1 row remaining|
|(18,7)|❌ FAIL|even height — convert stage|
|(2,7)|❌ FAIL|even height — convert stage|
|(19,7)|❌ FAIL|negative dimension (19 > 18 input rows)|

### v2 architecture probes — structural constraints (June 2026)

|Pattern|Result|Notes|
|---|---|---|
|Branching / Concatenate|❌ FAIL|convert — `re_lu_identity_conv followed by conv2d is not valid`|
|`DepthwiseConv2D` as first layer|❌ FAIL|quantize — must be Conv2D/Dense as first layer|
|`SeparableConv2D` as first layer|❌ FAIL|quantize — same constraint as DepthwiseConv2D|
|`(9,7)` spatio-temporal first layer|✅ PASS|Used in v2 — confirmed end-to-end|
|Full v2 architecture (akida_cnn_v2.py)|✅ PASS|probe3b equivalent, confirmed 3 June 2026|

---

## 2. Key findings

### B: Both pool orderings are valid

`Conv → Pool → ReLU` and `Conv → ReLU → Pool` both pass all three stages. MetaTF's `prepare_to_convert()` reorders/fuses blocks before the compat check runs. Both architectures use `Conv → Pool → ReLU` for consistency — this is preference, not constraint.

### Branching hard fails at convert

`Concatenate` merge points fail with `re_lu_identity_conv followed by conv2d is not valid`. Multi-scale / multi-branch architectures are completely off the table for AKD1000 v1.

### Kernel height must be odd (or 1)

Even kernel heights (2, 4, 6, 8, 10, 12, 14, 16, 18) fail at the convert stage with `kernel_width must be strictly positive and odd`. Height=1 is a valid special case. Maximum valid odd height for (18, 512, 1) input: 17 (leaves 1 row; 19 exceeds input).

### First layer must be Conv2D

`DepthwiseConv2D` and `SeparableConv2D` cannot be the first processing layer — the quantiser's InputConvolutional mapper requires `Conv2D` (or `Dense`) as entry point. `SeparableConv2D` is valid as a **non-first** layer (probe3 J confirmed).

---

## 3. Complete valid pattern reference

### Input block (first Conv → maps to InputConvolutional)

|Pattern|Valid|Notes|
|---|---|---|
|`InputConv2D(ch=1or3, w=odd) → Pool(pad=match) → ReLU(6)`|✓|Primary|
|`InputConv2D(ch=1or3, w=odd) → ReLU(6) → Pool`|✓|Both orderings valid|
|`InputConv2D(ch=1or3, w=odd) → ReLU(6)`|✓|No pool|
|`InputConv2D(ch=1or3, w=odd) → GAP → ReLU(6)`|✓|GAP variant|
|`DepthwiseConv2D` as first layer|✗|quantize fails — must be Conv2D|
|`SeparableConv2D` as first layer|✗|quantize fails — must be Conv2D|
|`InputConv2D(ch≠1,3)`|✗|Channel dim must be 1 or 3|
|`InputConv2D(weight_bits=2)`|✗|First conv: 4 or 8 bit only|
|`InputConv2D(kernel_height=even)`|✗|Kernel height must be odd or 1|

### Conv blocks (non-input)

|Pattern|Valid|Notes|
|---|---|---|
|`Conv(k=odd×odd) → Pool(pad=match) → ReLU(6)`|✓|**Primary — use this**|
|`Conv(k=odd×odd) → ReLU(6) → Pool`|✓|Conventional order valid|
|`Conv(k=odd×odd) → ReLU(6)`|✓|No pool|
|`Conv → BN → Pool → ReLU(6)`|✓|BN before pool (C)|
|`Conv → BN → ReLU(6)`|✓|BN no pool (D)|
|`Conv → GAP → ReLU(6)`|✓|GAP with activation (H)|
|`Conv → GAP`|✓|Terminal GAP, no ReLU needed|
|`SepConv → Pool → ReLU(6)`|✓|SeparableConv2D as non-first layer (J)|
|`SepConv → GAP → ReLU(6)`|✓|SeparableConv2D GAP variant|
|Branching / Concatenate|✗|convert fails — sequential only|
|`Conv → Pool → Pool`|✗|Two consecutive pools (L)|
|`Conv → GAP → Dense`|✗|GAP cannot precede Dense in v1 (G)|
|`Conv(activation='relu')` inline|✗|Inline activation (K)|
|`Conv → ReLU(unbounded)`|✗|Must use `ReLU(max_value=6)`|
|`Conv(weight_bits=8)` non-input|✗|8-bit only at input layer|
|`Conv(kernel_height=even)`|✗|Kernel height must be odd or 1|

### Between-block insertions

|Pattern|Valid|Notes|
|---|---|---|
|`... → ReLU(6) → Dropout(p) → Conv → ...`|✓|Confirmed (E)|
|`... → ReLU(6) → Reshape → Dense → ...`|✓|From source analysis|

### Head patterns

|Pattern|Valid|Notes|
|---|---|---|
|`Flatten → Dense → ReLU(6) → Dense(softmax)`|✓|**Confirmed (F, M, v2)**|
|`Flatten → Dense(softmax)`|✓|Simple head|
|`GAP → Dense`|✗|Hard fail in v1 (G)|
|`Dense(non-flat input)`|✗|Shape must be (N,) or (1,1,N)|

### Weight bit-widths

|Layer|Valid bits|Notes|
|---|---|---|
|First Conv (InputConv)|4, 8|8-bit = more precision at input|
|All other Conv|1, 2, 4|4-bit standard; 2-bit loses specificity|
|Dense|4|Standard|
|Activations|4|`per_tensor_activations=True` mandatory|

---

## 4. Confirmed architectures

### v1 — `akida_cnn.py` (temporal-only baseline)

**201,058 params | Confirmed compatible 31 May 2026**

```
Rescaling(1/255) → Conv2D(32,(1,7),s=(1,4),valid) → Pool(1,2,valid) → ReLU(6)
                 → Conv2D(64,(3,3),same)           → Pool(2,2,same)  → ReLU(6)
                 → Conv2D(32,(3,3),same)           → Pool(2,2,same)  → ReLU(6)
                 → Flatten → Dense(64) → ReLU(6) → Dense(2,softmax)
```

Block 1 kernel (1,7): temporal-only, treats 18 channels independently.

### v2 — `akida_cnn_v2.py` (spatio-temporal, primary model)

**137,314 params | Confirmed compatible 3 June 2026**

```
Rescaling(1/255) → Conv2D(32,(9,7),s=(1,4),valid) → Pool(1,2,valid) → ReLU(6)
                 → Conv2D(64,(3,3),same)            → Pool(2,2,same)  → ReLU(6)
                 → Conv2D(32,(3,3),same)            → Pool(2,2,same)  → ReLU(6)
                 → Flatten → Dense(64) → ReLU(6) → Dense(2,softmax)
```

Block 1 kernel (9,7): 9 electrodes × 7 timesteps — learns spatial co-activation patterns jointly with temporal dynamics. Biologically motivated: seizures propagate through contiguous electrode groups (6-12 electrodes typical in CHB-MIT). 32% fewer parameters than v1 — less overfit risk on limited seizure data.

**Patient-specific fine-tuning:** freeze conv1-3, train Dense head only (20 epochs, lr=1e-4). Adapts classification boundary to patient-specific ictal signatures while preserving general spatio-temporal feature extraction.

---

## 5. What you can and cannot do in future iterations

**Valid additions (all probed and confirmed):**

- `Dropout(p)` after any ReLU between blocks
- `BatchNormalization` as `Conv → BN → [Pool] → ReLU`
- `SeparableConv2D` as a non-first Conv block replacement
- Any odd kernel height ≤ 17 in the first layer
- Either pool ordering in any block

**Hard constraints (no workaround in v1):**

- No branching / Concatenate — sequential architectures only
- No GAP → Dense
- No DepthwiseConv2D or SeparableConv2D as first layer
- No even kernel heights
- No inline `activation='relu'`
- No unbounded ReLU
- No 8-bit weights on non-input layers
- Input channel dim must be 1 or 3

---

## 6. API quick-reference

```python
import tf_keras as keras                         # ALWAYS — never tensorflow.keras
from cnn2snn import (convert, check_model_compatibility,
                     set_akida_version, AkidaVersion)
from quantizeml.models import quantize, QuantizationParams

qparams = QuantizationParams(
    input_weight_bits=8,             # first conv: 4 or 8 only
    weight_bits=4,                   # other convs: 1, 2, or 4
    activation_bits=4,               # NOT activ_bits= (silently ignored)
    per_tensor_activations=True      # MANDATORY for v1 hardware
)

# Compat check — float model only, returns None, use try/except
with set_akida_version(AkidaVersion.v1):
    check_model_compatibility(float_model)

# Quantize — ONCE on float model
q_model = quantize(float_model, qparams=qparams, samples=cal_samples)

# Convert — no input_scaling= (deprecated)
with set_akida_version(AkidaVersion.v1):
    ak_model = convert(q_model, file_path='model.fbz')

# Inference
loaded = akida.Model('model.fbz')
labels = np.argmax(loaded.predict(X).squeeze(), axis=1)  # (N,1,1,C) → argmax
print(loaded.statistics)                                  # Statistics — print directly
```

---

## 7. Complete constraint table (23 entries)

| #   | Constraint                                                                                                                                               | Source                             |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 1   | `import tf_keras as keras` — never `from tensorflow import keras`                                                                                        | Phase 1b                           |
| 2   | `padding='valid'` on first Conv2D — quantizeml 1.2.3 strips `same`                                                                                       | Phase 1b                           |
| 3   | `keras.layers.ReLU(max_value=6)` as separate layer — NOT `activation='relu'`                                                                             | probe3 K                           |
| 4   | `QuantizationParams(activation_bits=4)` — NOT `activ_bits=` (silently ignored)                                                                           | Phase 1b                           |
| 5   | `check_model_compatibility()` returns `None` — use try/except                                                                                            | Phase 1b                           |
| 6   | Call `check_model_compatibility()` on **float model only**                                                                                               | Phase 1b                           |
| 7   | `set_akida_version(AkidaVersion.v1)` on every `convert()` AND compat check                                                                               | Phase 1b                           |
| 8   | `convert(model)` — do NOT pass `input_scaling=` (deprecated)                                                                                             | Phase 1b                           |
| 9   | SNN `predict()` output shape is `(N,1,1,C)` — `squeeze()` then `argmax`                                                                                  | Phase 1b                           |
| 10  | `loaded.statistics` is a Statistics object — `print()` directly                                                                                          | Phase 1b                           |
| 11  | Both `Conv → Pool → ReLU` and `Conv → ReLU → Pool` are valid                                                                                             | probe3 A,B                         |
| 12  | MaxPool padding must match its Conv padding in the same block                                                                                            | Source analysis                    |
| 13  | No two consecutive MaxPool layers                                                                                                                        | probe3 L                           |
| 14  | GAP cannot be followed by Dense in AKD1000 v1                                                                                                            | probe3 G                           |
| 15  | Valid head: `Flatten → Dense → ReLU → Dense(softmax)`                                                                                                    | probe3 F,M                         |
| 16  | `per_tensor_activations=True` MANDATORY in QuantizationParams                                                                                            | Phase 1b                           |
| 17  | Converted models saved as `.fbz` not `.h5`                                                                                                               | Phase 1b                           |
| 18  | `input_shape[-1]` must be 1 or 3 for first Conv (InputConvolutional)                                                                                     | Source analysis                    |
| 19  | First Conv `weight_bits`: 4 or 8 only. All others: 1, 2, or 4                                                                                            | Source analysis                    |
| 20  | `quantize()` called ONCE on float model — requantising raises error                                                                                      | probe2 post-mortem                 |
| 21  | Calibration data shape must match model `input_shape` exactly                                                                                            | probe3b                            |
| 22  | Kernel height must be odd (or 1) — even heights fail at convert stage                                                                                    | v2 probing                         |
| 23  | No branching/Concatenate — AKD1000 v1 is strictly sequential                                                                                             | v2 probing                         |
| 24  | `dilation_rate != 1` unsupported on any Conv2D — fails at `check_model_compatibility()` (compat stage, not convert), regardless of which axis is dilated | probe_dilated_conv.py, 4 July 2026 |

---

## 8. Probe history

| Probe                | Purpose                                                                                                                                                  | Outcome                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| probe1               | MetaTF source extraction: ConvBlockConverterV1, DenseBlockConverterV1, InputConvBlockConverterV1, compatibility_checks.py                                | Source-level constraint map                              |
| probe2               | Pattern testing (13 patterns)                                                                                                                            | All failed — requantize bug in script; discarded         |
| probe3               | Corrected pattern testing (13 patterns)                                                                                                                  | 9/13 pass; test M false-negative (wrong dummy shape)     |
| probe3b              | Test M with correct dummy shape (18,512,1)                                                                                                               | PASS — v1 architecture confirmed end-to-end               |
| v2 probes A–B        | Branching (Concatenate), DepthwiseConv2D first layer                                                                                                     | Both FAIL — sequential only, Conv2D first layer required |
| v2 probes C–E        | Spatial kernels: (18,1), (18,7), SeparableConv2D first layer                                                                                              | All FAIL — even height or non-Conv2D first layer          |
| v2 kernel probes     | (1,7),(3,7),(5,7),(7,7),(9,7),(17,7),(18,7),(2,7),(19,7)                                                                                                  | Odd heights pass; even heights fail; >18 negative dim     |
| v2 candidate probes  | (17,7),(9,1)→(1,7),(9,7),(3,7),(5,7),(7,7) full pipeline                                                                                                 | All 6 PASS — (9,7) selected for v2 on biological grounds  |
| v2 full architecture | akida_cnn_v2.py end-to-end                                                                                                                                | PASS — confirmed 3 June 2026                              |
| sandbox probe        | `dilation_rate != 1` unsupported on any Conv2D — fails at `check_model_compatibility()` (compat stage, not convert), regardless of which axis is dilated | probe_dilated_conv.py, 4 July 2026                        |
