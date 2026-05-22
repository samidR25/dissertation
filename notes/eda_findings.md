# EDA Findings — chb01

## Signal quality
- 256 Hz confirmed
- All 18/18 common channels present
- Raw signal: clean, no dead channels, FP1-F7 shows occasional blink artefacts (normal)

## Frequency content
- 1/f slope present — confirms real EEG
- Alpha peak visible ~10 Hz
- 60 Hz mains spike confirmed → notch filter correctly targeted
- Harmonic spikes at ~30, 48, 77 Hz → removed by 40 Hz low-pass cutoff

## Class balance (Table 1)
- Total recording: 40.55 hours
- Seizure time: 442.0s (0.30%)
- Imbalance ratio: 1:329
- Total seizures: 7
- Primary metric: SENSITIVITY (not accuracy — 98% trivially achieved by all-negative)

## Files
- 42 EDF files (chb01_28, 35, 44, 45 absent — normal for CHB-MIT)
- Short files: chb01_20 (~44 min), chb01_26 (~39 min), chb01_27 (10 min)
