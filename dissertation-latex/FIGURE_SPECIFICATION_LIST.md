# Figure Specification List

For handing to another chat/tool to generate. Organised by dissertation section. Each entry gives: the expected filename (matches the `\figplaceholder` calls already in the .tex files), what it should show, and why it's needed.

**Note on assets that may already exist:** several filenames below (`lopo_sensitivity_bimodal_hist.png`, `lopo_per_patient_bars.png`, `lopo_sensitivity_vs_fphr_scatter.png`, `c2_vs_lopo_comparison.png`, `chb13_convergence.png`, `pipeline_architecture.png`, `collapse_diagnostic_illustration.png`, and the `eeg_chb*` waveform images) appear to already exist in the project's file store from earlier sessions. Worth checking those first — some may only need re-verifying against the current (post-restructuring) numbers rather than regenerating from scratch, since the pool6-minus-X/pool7 reordering happened after some of these were likely made.

---

## Chapter 3 — Methodology (net new, not yet generated)

**1. `eeg_electrode_montage.png`**
A standard 10-20 system EEG electrode placement diagram (top-down view of the scalp). The 18 channels used throughout this project should be clearly marked/highlighted; any excluded channels should be visibly crossed out or greyed out, so a reader can see at a glance which scalp regions the models do and do not have access to. Used in §3.1 (Dataset and Preprocessing).

**2. `chronological_split_timeline.png`** *(not yet placeholdered in the .tex — recommend adding)*
A simple horizontal timeline bar per patient, divided into three coloured segments (70% train / 15% validation / 15% test, left to right in chronological order), making the "earlier portion is training, later portion is test" logic immediately visual rather than only described in prose. Would strengthen §3.1 alongside the code snippet already there.

**3. `pipeline_architecture_full.png`**
A comprehensive end-to-end pipeline diagram: raw scalp EEG input → preprocessing/chronological split → `seizure_cnn_v2` architecture → w4a4 quantisation → ANN-to-SNN conversion (`cnn2snn`) → deployment on AKD1000 v1 silicon. Each stage should be annotated with the relevant section number (§3.1–§3.5) so the diagram doubles as a chapter roadmap. Used in §3.2 (Model Architecture).

**4. `cnn_architecture_layers.png`** *(recommend adding)*
A layer-by-layer diagram of `seizure_cnn_v2` itself: the three repeated Conv→BN→Pool→ReLU blocks, then Flatten→Dense(64)→Dense(2, softmax), with approximate parameter counts per block if easy to include. This is a narrower, more detailed companion to the full pipeline diagram above — useful right where the architecture is first described in prose (§3.2.1).

**5. `ann_to_snn_conversion_diagram.png`** *(recommend adding)*
A conceptual before/after diagram showing a ReLU unit's continuous activation value on one side, mapped to a LIF neuron's spike train (several spikes over a short time window) on the other, visually illustrating the rate-coding relationship described by Equation 2.1 in §2.3/§3.2.3. Would make the ANN→SNN conversion concept far more concrete than the equation alone.

**6. `lif_membrane_potential_trace.png`** *(recommend adding)*
A plot of LIF membrane voltage over time, showing the characteristic sawtooth pattern: gradual integration toward threshold, a spike at threshold crossing, then reset. (This may already exist as `/tmp/lif_demo.png` if the demo script in the project's foundational notes was run — worth checking before regenerating.) Pairs directly with Equation 2.1.

**7. `training_regime_pool_composition.png`** *(recommend adding)*
A simple set-diagram or table-style graphic showing which patients belong to which training regime: C1 (chb01/02/05), pool7 (chb01/02/05/06/07/09/20), pool6-minus-X (pool7 minus one patient, repeated per patient), and C2 (chb10/13/15/16, fine-tuned individually). Would resolve a lot of the "which patients are in which pool" bookkeeping the reader currently has to hold in their head across §3.3.

**8. `hardware_setup_photo.png`**
An actual photograph of the working physical deployment: Raspberry Pi 5, AKD1000 v1 M.2 card and HAT+, and the GPIO-mounted cooling fan soldered in place. Used in §3.5 (Hardware Deployment).

**9. `rs_pro_bench_supply_photo.png`**
A photograph of the RS PRO RS-3005P bench DC power supply, ideally shown connected to the Pi 5's USB-C input via the bench-supply-to-USB-C adapter, so the measurement setup is visually verifiable. Used in §3.5.2 (Power Measurement Methodology).

**10. `power_measurement_circuit_diagram.png`** *(recommend adding)*
A simple block diagram showing the Pi 5's 5V USB-C input, the internally-derived 3.3V rail feeding the HAT+/AKD1000 card, and the single external measurement point — visually justifying, in one glance, why the bench measurement is necessarily system-level rather than chip-isolated (the point made in prose in §3.5.2).

## Chapter 4 — Results

**11. `lopo_sensitivity_bimodal_hist.png`**
Histogram of the 8 collapse-passing LOPO fold event sensitivities, showing the bimodal clustering near 0.0 and 1.0. Used in §4.1.2.

**12. `lopo_per_patient_bars.png`**
Per-patient bar chart of LOPO event sensitivity and FP/hr, grouped/coloured by collapse status (PASS vs FAIL). Used in §4.1.3.

**13. `lopo_sensitivity_vs_fphr_scatter.png`**
Scatter plot of event sensitivity against FP/hr across all 15 LOPO folds, coloured by collapse status. Used in §4.1.3.

**14. `chb13_convergence.png`**
A figure showing chb13's performance under DANN, CORAL, self-supervised pretraining, and Candidate G side by side (e.g. small multiples or a grouped bar chart), visually reinforcing the four-way convergence finding. Used in §4.4.

**15. `power_measurement_bar_chart.png`** *(recommend adding)*
A simple bar chart of the three power-measurement trials (5.57W, 5.68W, 5.63W) alongside their mean, as a visual companion to Table 4.x in §4.5.1 — mostly for readability given how central this figure is to the hardware chapter.

## Appendix

**16. `c2_vs_lopo_comparison.png`**
Per-patient C2 fine-tuning results compared against the corresponding LOPO zero-shot result for the same patient, where available. Used in Appendix §A.2.

## Additional waveform/illustrative figures (per your note on visually aiding understanding)

**17. `eeg_seizure_vs_baseline_example.png`** *(may already exist as separate `eeg_chb10_seizure.png` / `eeg_chb10_baseline.png` files)*
A side-by-side raw EEG waveform comparison: a short segment of genuine seizure activity next to a short segment of quiet interictal background, from the same patient, same channel, same y-axis scale — giving a reader an immediate visual sense of what the model is actually being asked to distinguish (ties directly to §2.1's discussion of ictal waveform morphology).

**18. `eeg_chb13_vs_chb10_comparison.png`** *(may already exist)*
A comparison of seizure-event waveform morphology between chb10 (a patient the pipeline handles well) and chb13 (the project's consistently hardest patient across every method), to give visual grounding to the repeated "chb13 is structurally hard" claim made throughout Chapters 4–5.

**19. `collapse_diagnostic_illustration.png`** *(may already exist)*
A visual illustration of the collapse diagnostic's three failure conditions (§3.4.3): e.g. three small panels showing (a) a normal, discriminating prediction trace, (b) an over-firing collapse (near-permanent block), (c) an under-firing collapse (almost no detections) — makes the three numeric thresholds immediately intuitive alongside the code snippet already in that section.

---

**Suggested priority if generating in batches:** items 3, 4, 7 (pipeline/architecture/pool-composition diagrams) and 1 (EEG montage) are the highest-value additions for reader comprehension and should come first; items 8–10 (hardware photos/diagram) next, since they're currently the least self-explanatory sections without a visual; the Results-chapter data plots (11–15) last, since real numbers already exist for all of them and generation there is mostly a plotting exercise against files already in `results/`.
