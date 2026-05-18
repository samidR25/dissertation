# Neuromorphic Seizure Detection — MSc Dissertation

**Stack:** Python 3.11 · TF 2.19.0 · tf-keras · akida 2.19.1 · cnn2snn 2.19.1
**Dataset:** CHB-MIT Scalp EEG (PhysioNet)
**Hardware:** RPi 5 + AKD1000 M.2 + M.2 HAT+
**Deadline:** 10 September 2026

## Setup
    python3.11 -m venv ~/venvs/akida_env
    source ~/venvs/akida_env/bin/activate
    pip install -r requirements.txt

## CRITICAL import rule
    ALWAYS:  import tf_keras as keras
    NEVER:   from tensorflow import keras   (imports Keras 3)
    NEVER:   import keras                   (imports Keras 3)

## Run order
    1. python3 code/preprocessing/eda.py
    2. python3 code/preprocessing/preprocess.py
    3. python3 code/preprocessing/build_dataset.py
    4. python3 code/models/smoke_test.py --samples 200 --epochs 2 --gate 1
    5. python3 code/models/train_baseline.py
    6. python3 code/models/convert_to_snn.py
    7. [RPi 5] python3 code/hardware/run_on_akida.py
    8. [RPi 5] python3 code/deployment/realtime_inference.py
