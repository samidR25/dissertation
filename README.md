# Neuromorphic EEG Seizure Detection

**MSc Advanced Computer Science Dissertation — Cardiff University, 2026**  
Supervisor: Dr. Nick Pham, Cardiff Agile CPS Lab

End-to-end pipeline for epileptic seizure detection on neuromorphic silicon: from raw scalp EEG through a quantised spatio-temporal CNN, converted to a Spiking Neural Network and deployed on a BrainChip AKD1000 neuromorphic processor at **~1 mW active power**.

---

## Hardware

![RPi 5 + AKD1000 M.2 hardware setup](assets/hardware_setup.jpeg)

*Raspberry Pi 5 with BrainChip AKD1000 REV2.0 M.2 card via PCIe M.2 HAT+*

---

## Key results (Phase 2b — hardware inference, June 2026)

| Metric | Value | Notes |
|--------|-------|-------|
| Inference latency | **0.91 ms / window** | Measured on AKD1000 silicon |
| Throughput | **~1,100 windows/s** | 2s EEG windows |
| Energy / inference | **0.91 µJ** | At 1 mW active power (datasheet) |
| AKD1000 active power | **~1 mW** | vs 15–50 mW on conventional MCU |
| Within-patient F1 (chb01) | **0.97** | SNN on AKD1000 simulator |
| Cross-patient sensitivity (chb03) | **0.46** | Held-out patient, hardware inference |
| Cross-patient specificity (chb01) | **0.98** | 500-window hardware evaluation |

> Phase 2c (multi-patient training + gradual unfreezing + majority vote post-processing) is in progress — targeting chb03 sensitivity ≥ 0.65.

---

## Pipeline overview

```
CHB-MIT Scalp EEG (PhysioNet)
         │
         ▼
  preprocess.py          — bandpass 0.5–40 Hz, 18-channel pick,
                           2s windows @ 50% overlap, rate-code → spikes
         │
         ▼
  build_dataset.py       — chronological 70/15/15 split,
                           undersample + SMOTE on train only
         │
         ▼
  train_baseline.py      — spatio-temporal CNN v2 (9,7) kernel,
                           137k params, w4a4 quantisation
         │
         ▼
  convert_to_snn.py      — ANN → SNN via MetaTF cnn2snn,
                           AkidaVersion.v1 target
         │
         ▼
  run_on_akida.py        — hardware inference on AKD1000 M.2,
  [Raspberry Pi 5]         power / latency / clinical metrics
```

---

## Architecture

**v2 spatio-temporal CNN** (primary model, AKD1000 v1 compatible):

```
Input (18, 512, 1)               — 18 EEG channels × 512 samples
  │
  ├─ Rescaling(1/255)            — [0,255] → [0,1] normalisation
  ├─ Conv2D (9,7) × 32           — spatio-temporal: 9 channels × 7 timesteps
  ├─ MaxPool (2,4)
  ├─ ReLU(max_value=6)
  ├─ Conv2D (3,3) × 64
  ├─ MaxPool (2,2)
  ├─ ReLU(max_value=6)
  ├─ Conv2D (3,3) × 32
  ├─ ReLU(max_value=6)
  ├─ Flatten
  ├─ Dense(64)
  ├─ ReLU(max_value=6)
  └─ Dense(2, softmax)           — [non-seizure, seizure]

Params: 137,314  |  Quantisation: w4a4  |  Spike sparsity: 90.2%
```

The (9,7) kernel is biologically motivated: seizures propagate through contiguous electrode groups (typically 6–12 channels). A 9-channel kernel covers ~50% of the 18-channel array, capturing ictal propagation extent while preserving spatial position for downstream layers.

---

## Neuromorphic hardware constraints discovered

During development, **23 previously undocumented AKD1000 v1 hardware constraints** were identified through systematic architectural probing — none are in BrainChip's public documentation. Selected critical constraints:

- Kernel heights must be **odd** (1, 3, 5, 7, 9 ... 17) — even heights fail silently at convert
- **No branching** (Concatenate, Add, Multiply) — sequential architectures only
- Block ordering must be **Conv → Pool → ReLU**, not Conv → ReLU → Pool
- **No DepthwiseConv2D or SeparableConv2D** as first layer
- `padding='valid'` required on first Conv2D (quantizeml 1.2.3 bug with `'same'`)
- `ReLU(max_value=6)` must be a **separate layer**, not `activation='relu'` inline
- GAP cannot be followed by any Dense layer — use Flatten instead
- `per_tensor_activations=True` mandatory in QuantizationParams
- `numpy >= 2.0` required (not `< 2.0` as stated in some older docs)

See [`AKD1000_v1_Architecture_Constraints.md`](AKD1000_v1_Architecture_Constraints.md) for the full list.

---

## Dataset

**CHB-MIT Scalp EEG Database** (PhysioNet) — paediatric patients, Children's Hospital Boston.

| Patient | Seizures | Test seizures | Primary use |
|---------|----------|---------------|-------------|
| chb01 | 7 | 0 (chronological split) | Within-patient baseline |
| chb02 | 3 | 0 | Cross-patient training |
| chb03 | 7 | 167 windows | **Primary cross-patient evaluation** |
| chb05 | 5 | 0 | Cross-patient training |

- Window: 2.0s @ 256 Hz = 512 samples, 50% overlap
- Channels: 18 (common set, positions 0–17)
- Class imbalance: 1 : 329 (seizure : non-seizure)
- Handled via: chronological split → undersample → SMOTE on train only

---

## Stack

| Component | Version |
|-----------|---------|
| Python | 3.11.x |
| TensorFlow | 2.19.1 |
| tf-keras | 2.19.0 |
| akida | 2.19.1 |
| cnn2snn | 2.19.1 |
| quantizeml | 1.2.3 |
| numpy | 2.1.3 |
| OS (dev) | Ubuntu 22.04 (WSL2) |
| OS (deploy) | RPi OS Bookworm 64-bit |

---

## Setup

```bash
python3.11 -m venv ~/akida_env
source ~/akida_env/bin/activate
pip install -r requirements.txt
pip install -e .
```

> ⚠️ **Critical import rule** — MetaTF requires Keras 2, not Keras 3:
> ```python
> import tf_keras as keras          # ✅ always
> from tensorflow import keras      # ❌ imports Keras 3, breaks MetaTF
> import keras                      # ❌ imports Keras 3 standalone
> ```

---

## Run order

```bash
# 1. Preprocessing (WSL2)
python3 src/preprocessing/preprocess.py --patient chb01
python3 src/preprocessing/build_dataset.py --patient chb01

# 2. Smoke test — must pass before full training
python3 src/models/smoke_test.py --gate 2

# 3. Training (WSL2, RTX 3060)
python3 src/models/train_baseline.py --patient chb01 --model-version 2 --class-weight 1.5

# 4. SNN conversion
python3 src/models/convert_to_snn.py --patient chb01 --model-version 2

# 5. Hardware inference (Raspberry Pi 5)
python3 src/hardware/run_on_akida.py --patient chb01 --n-eval 500
```

---

## Repository structure

```
dissertation/
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py          — EEG windowing, filtering, spike encoding
│   │   └── build_dataset.py       — chronological split, SMOTE balancing
│   ├── models/
│   │   ├── akida_cnn.py           — v1 architecture (1,7) temporal-only
│   │   ├── akida_cnn_v2.py        — v2 architecture (9,7) spatio-temporal ← primary
│   │   ├── train_baseline.py      — training + patient fine-tuning
│   │   ├── convert_to_snn.py      — ANN→SNN conversion, AkidaVersion.v1
│   │   └── smoke_test.py          — gates 1/2/3 compatibility checks
│   └── hardware/
│       └── run_on_akida.py        — chip inference, power/latency measurement
├── data/
│   ├── raw/chbmit/                — CHB-MIT EDF files (not tracked — see PhysioNet)
│   └── processed/                 — .npz datasets (Git LFS)
├── results/                       — trained models (.h5, .fbz), metrics (.json)
├── assets/                        — hardware photos, diagrams
└── writing/                       — dissertation chapters (in progress)
```

---

## Current status

| Phase | Status |
|-------|--------|
| Phase 1 — Preprocessing (chb01–05) | ✅ Complete |
| Phase 2a — Model development + ablations | ✅ Complete |
| Phase 2b — Hardware deployment (AKD1000) | ✅ Complete |
| Phase 2c — Cross-patient generalisation | 🔄 In progress |
| Phase 2d — Frequency features + LOPO | ⬜ Planned |
| Phase 3 — Dissertation write-up | ⬜ Planned |

---

## References

- Shoeb, A. & Guttag, J. (2010). Application of machine learning to epileptic seizure onset detection. *ICML 2010*. [CHB-MIT dataset]
- Wu, Y. et al. (2018). Spatio-temporal backpropagation for training high-performance spiking neural networks. *Frontiers in Neuroscience*, 12, 331.
- BrainChip Inc. (2023). AKD1000 SoC Product Brief. [BrainChip technical documentation]
- Chawla, N.V. et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.

---

*Deadline: 10 September 2026 · Cardiff University · Supervised by Dr. Nhat Pham*
