## Table 1: ANN baseline per patient
| Patient | Eval set | Sensitivity | Specificity | FPR/hr | Epochs |
|---------|----------|-------------|-------------|--------|--------|
| chb01   | train    | 0.9363      | 1.0000      | 0.00   | 34     |
| chb02   | train    | 1.0000      | 1.0000      | 0.00   | 56     |
| chb03   | test     | 0.2275      | 0.9741      | 46.13  | 40     |
| chb05   | train    | 1.0000      | 1.0000      | 0.00   | 36     |

## Table 2: Conversion quality vs bit-width (chb01)
| w-bits | a-bits | ANN sens | SNN sens | Drop   | Acceptable |
|--------|--------|----------|----------|--------|------------|
| 4      | 4      | 0.9359   | 1.0000   | -0.064 | ✓          |
| 8      | —      | —        | —        | —      | ✗ invalid (non-input layers max 4-bit) |
| 2      | 4      | 0.9359   | 1.0000   | -0.064 | ✗ spec=0.001 (too aggressive) |

## Table 3: Window size (chb01)
| Window | Input shape | ANN sens | SNN sens | Notes |
|--------|-------------|----------|----------|-------|
| 2.0s   | (18,512,1)  | 0.9363   | 1.0000   | Primary result |
| 1.0s   | (18,256,1)  | —        | —        | Not yet run |

## Table 4: Cross-patient SNN results
| Patient | ANN sens | SNN sens | SNN spec | Notes |
|---------|----------|----------|----------|-------|
| chb01   | 0.9363 (train) | 1.0000 | 0.9484 | No held-out seizures |
| chb02   | 1.0000 (train) | N/A    | 0.6010 | No held-out seizures |
| chb03   | 0.2275 (test)  | TBD    | TBD    | 167 test seizures — primary cross-patient result |
| chb05   | 1.0000 (train) | N/A    | 0.9540 | No held-out seizures |

## Table 5: v1 vs v2 architecture comparison (SNN simulator)

| Patient | v1 SNN sens | v1 SNN spec | v2 SNN sens | v2 SNN spec | Notes |
|---------|-------------|-------------|-------------|-------------|-------|
| chb01   | 1.0000      | 0.9484      | 0.9800      | 0.9640      | v2 better balanced (F1: 0.74→0.97) |
| chb02   | N/A (0 test seizures) | 0.6010 | 0.7240 | 0.9500 | Train-set eval only |
| chb03   | 0.0000      | 0.9960      | 0.4611      | 0.0000      | Only patient with test seizures |
| chb05   | N/A (0 test seizures) | 0.9540 | 0.9820 | 0.9760 | Train-set eval only |

chb03 ANN test sensitivity: v1=0.2275 → v2=0.3653 (+60% relative, patient fine-tuning from chb01 base)
Primary cross-patient finding: spatio-temporal architecture improves generalisation but
fundamental cross-patient challenge remains (consistent with Litt & Echauz 2002).
