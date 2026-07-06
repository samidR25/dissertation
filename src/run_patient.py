"""
src/run_patient.py
===================
Gate 1c — thin orchestration entry point (Phase 3 Session 2).

Runs preprocess -> build_dataset -> train -> convert -> eval as gated
stages, driven by a single PipelineConfig, calling the EXISTING scripts
via subprocess (wrap, don't replace — no stage's internals live here).
Resumable: each stage is skipped if its expected output (and manifest,
where one exists) is already present, unless --force is given.

NOTE: this calls train_baseline.py with TODAY's CLI (--finetune-from +
--gradual-unfreeze), not a --freeze-depth flag — that doesn't exist until
Gate 2. cfg.freeze_depth is defined (forward-looking, from Gate 1a) but
not yet wired through here.

Usage:
    python3 src/run_patient.py --patient chb10 --finetune-from multi \
        --gradual-unfreeze --seed 123

    python3 src/run_patient.py --patient chb10 --stop-after build_dataset

    python3 src/run_patient.py --patient chb10 --stop-after build_dataset --force
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, '.')
from src.config import PipelineConfig
from src.manifest import load_manifest

STAGES = ['preprocess', 'build_dataset', 'train', 'convert', 'eval']


def _run(cmd, label):
    print(f"\n{'='*70}\n[run_patient] {label}\n  $ {' '.join(cmd)}\n{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[run_patient] STAGE FAILED: {label} (exit {result.returncode})")


def stage_preprocess(cfg, args):
    x_path = f'data/processed/{cfg.patient}_X.npy'
    y_path = f'data/processed/{cfg.patient}_y.npy'
    if not args.force and os.path.exists(x_path) and os.path.exists(y_path):
        print(f"[run_patient] SKIP preprocess — {x_path} already exists")
        return
    _run(['python3', 'src/preprocessing/preprocess.py', '--patient', cfg.patient],
         label=f"preprocess {cfg.patient}")


def stage_build_dataset(cfg, args):
    npz_path = f'data/processed/{cfg.patient}_dataset_ann.npz'
    manifest = load_manifest(npz_path, required=False)
    if not args.force and os.path.exists(npz_path) and manifest is not None:
        print(f"[run_patient] SKIP build_dataset — {npz_path} + manifest already exist")
        return
    cmd = ['python3', 'src/preprocessing/build_dataset.py',
           '--patient', cfg.patient,
           '--train-frac', str(cfg.split_ratios[0]),
           '--val-frac', str(cfg.split_ratios[1])]
    _run(cmd, label=f"build_dataset {cfg.patient}")


def stage_train(cfg, args):
    ckpt_path = f'results/best_ann_{cfg.patient}_v{cfg.model_version}.h5'
    manifest = load_manifest(ckpt_path, required=False)
    if not args.force and os.path.exists(ckpt_path) and manifest is not None:
        print(f"[run_patient] SKIP train — {ckpt_path} + manifest already exist")
        return

    if cfg.finetune_from:
        # Gate 2c pre-check: personalised training must use the patient's
        # OWN scaler, never the pool's. Nothing else enforces this yet.
        data_manifest = load_manifest(
            f'data/processed/{cfg.patient}_dataset_ann.npz', required=True)
        expected_scaler_path = f'data/processed/{cfg.patient}_scaler.json'
        if data_manifest.get('scaler_path') != expected_scaler_path:
            sys.exit(
                f"ERROR: {cfg.patient}'s dataset manifest records scaler_path="
                f"{data_manifest.get('scaler_path')}, expected "
                f"{expected_scaler_path}. Personalised training must use the "
                "patient's own scaler (Gate 2c) — refusing to proceed."
            )

    cmd = ['python3', 'src/models/train_baseline.py',
           '--patient', cfg.patient,
           '--model-version', str(cfg.model_version),
           '--seed', str(cfg.seed)]
    if cfg.finetune_from:
        cmd += ['--finetune-from', cfg.finetune_from]
        if args.gradual_unfreeze:
            cmd += ['--gradual-unfreeze']
    _run(cmd, label=f"train {cfg.patient}")


def stage_convert(cfg, args):
    fbz_path = (f'results/seizure_model_{cfg.patient}_v{cfg.model_version}'
                f'_w{cfg.bit_widths[0]}a{cfg.bit_widths[1]}.fbz')
    manifest = load_manifest(fbz_path, required=False)
    if not args.force and os.path.exists(fbz_path) and manifest is not None:
        print(f"[run_patient] SKIP convert — {fbz_path} + manifest already exist")
        return
    cmd = ['python3', 'src/models/convert_to_snn.py',
           '--patient', cfg.patient,
           '--model-version', str(cfg.model_version),
           '--w-bits', str(cfg.bit_widths[0]),
           '--a-bits', str(cfg.bit_widths[1])]
    _run(cmd, label=f"convert {cfg.patient}")


def stage_eval(cfg, args):
    fbz_path = (f'results/seizure_model_{cfg.patient}_v{cfg.model_version}'
                f'_w{cfg.bit_widths[0]}a{cfg.bit_widths[1]}.fbz')
    out_path = (f"results/event_results_seizure_model_{cfg.patient}"
                f"_v{cfg.model_version}_w{cfg.bit_widths[0]}a{cfg.bit_widths[1]}"
                f"_on_{cfg.patient}.json")
    if not args.force and os.path.exists(out_path):
        print(f"[run_patient] SKIP eval — {out_path} already exists")
        return
    cmd = ['python3', 'src/evaluation/eval_event_level.py',
           '--fbz', fbz_path,
           '--eval-patient', cfg.patient,
           '--spike-threshold', str(cfg.threshold),
           '--gap-tolerance', str(cfg.gap_tolerance_s),
           '--detection-fraction', str(cfg.detection_fraction),
           '--min-sustained', str(cfg.min_sustained_windows)]
    if cfg.scaler_source:
        cmd += ['--scaler-source', cfg.scaler_source]
    _run(cmd, label=f"eval {cfg.patient}")


STAGE_FNS = {
    'preprocess': stage_preprocess,
    'build_dataset': stage_build_dataset,
    'train': stage_train,
    'convert': stage_convert,
    'eval': stage_eval,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-version', type=int, default=2)
    parser.add_argument('--finetune-from', default=None)
    parser.add_argument('--gradual-unfreeze', action='store_true')
    parser.add_argument('--scaler-source', default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--w-bits', type=int, default=4)
    parser.add_argument('--a-bits', type=int, default=4)
    parser.add_argument('--stop-after', choices=STAGES, default='eval')
    parser.add_argument('--force', action='store_true',
                        help="Re-run every stage even if valid output exists")
    args = parser.parse_args()

    cfg = PipelineConfig(
        patient=args.patient, seed=args.seed, model_version=args.model_version,
        finetune_from=args.finetune_from, scaler_source=args.scaler_source,
        threshold=args.threshold, bit_widths=(args.w_bits, args.a_bits),
    )
    print(cfg.summary())

    stop_idx = STAGES.index(args.stop_after)
    for stage_name in STAGES[:stop_idx + 1]:
        STAGE_FNS[stage_name](cfg, args)

    print(f"\n[run_patient] Done — stopped after '{args.stop_after}'.")
