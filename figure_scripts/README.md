# Dissertation figure-generation scripts

These reproduce the 9 net-new figures I generated (items 1, 2, 3, 4, 5, 6, 7, 10, 15,
17 from the figure spec list). They are plain `matplotlib`/`Pillow` scripts — no
TensorFlow, akida, or cnn2snn dependency — so they'll run in your `akida_env` conda
environment as-is, or in any plain Python 3 environment.

## Setup (WSL2, using your existing akida_env)

```bash
conda activate akida_env
pip install matplotlib numpy pillow --break-system-packages   # skip any already installed
```

(If you'd rather not touch akida_env, a throwaway venv works too:
`python3 -m venv figvenv && source figvenv/bin/activate && pip install matplotlib numpy pillow`)

## Running

Each script is self-contained and writes its PNG into the current working directory.
Run them from inside this folder:

```bash
cd figure_scripts
python3 01_eeg_electrode_montage.py
python3 02_chronological_split_timeline.py
python3 03_pipeline_architecture_full.py
python3 04_cnn_architecture_layers.py
python3 05_ann_to_snn_conversion_diagram.py
python3 06_lif_membrane_potential_trace.py
python3 07_training_regime_pool_composition.py
python3 10_power_measurement_circuit_diagram.py
python3 15_power_measurement_bar_chart.py
```

`17_eeg_seizure_vs_baseline_example.py` is different: it doesn't generate data, it
combines your **existing** `eeg_chb10_baseline.png` and `eeg_chb10_seizure.png` files
side by side. Edit the two path variables at the top of that script to point at
wherever those two files actually live on your machine before running it.

## What each script assumes / where its numbers come from — check these against your own source of truth

- **01 (electrode montage):** channel list is hard-coded from your
  `03_DATASET_AND_EEG_md.md` `COMMON_CHANNELS` list (the 18 bipolar pairs). The
  4 "excluded, patient-specific extra" electrodes (FT9/FT10/T1/T2) are an
  **illustrative guess** at what CHB-MIT's non-common extra channels typically are —
  I did not have your per-patient raw channel lists, so verify these 4 names/positions
  against an actual `chb01_01.edf` channel dump before trusting this figure completely.
- **02 (chronological split):** uses your fixed `split_ratios = (0.70, 0.15, 0.15)`
  applied identically to all 15 patients — purely illustrative of the *ratio*, not
  actual recording durations per patient.
- **03 (pipeline diagram):** stage labels + section numbers (§3.1–§3.5) taken directly
  from your dissertation's chapter structure. No underlying data, just layout — check
  the section numbers still match if you renumber the chapter.
- **04 (CNN layers):** block structure (3× Conv-BN-Pool-ReLU → Flatten → Dense(64) →
  Dense(2)) and the two parameter counts (~38,900 trunk / ~98,400 head) are taken
  verbatim from your Methodology text (Section 3.3.3) — not recomputed from your
  actual `model.summary()`. Worth cross-checking against a real summary() call.
- **05 (ANN→SNN diagram):** purely conceptual/illustrative — the activation value
  (0.72) and resulting spike count are arbitrary, chosen only to make the rate-coding
  relationship visually clear. Not derived from a real unit's activation.
- **06 (LIF trace):** a toy LIF simulation (`tau=0.02`, `R=1.0`, `I=1.6`, `Vth=1.0`) —
  parameters chosen only to produce a readable number of spikes over 500ms, not taken
  from your actual conversion toolchain's neuron parameters.
- **07 (pool composition):** patient-to-regime membership (C1/pool7/C2 columns) is
  taken directly from Section 3.3 of your dissertation text.
- **10 (power circuit diagram):** purely conceptual block diagram from your Section
  3.5.2 prose (Pi 5 5V rail → internal 3.3V → HAT+/AKD1000; fan on GPIO). No numeric
  data.
- **15 (power bar chart):** uses your three actual measured trial values (5.57 W,
  5.68 W, 5.63 W) from Table 4.5.
- **17 (EEG combination):** no new data — just recombines your two existing PNGs.

If you regenerate 01, 04, or 06 with your actual data (channel dump, model.summary(),
or real neuron params), they'll be strictly better than what I produced from text
descriptions alone.
