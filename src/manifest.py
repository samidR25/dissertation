"""
src/manifest.py
================
Manifest sidecar system (Phase 3 Session 2, Gate 1b).

Every artifact a stage script writes (a built dataset npz, a trained
checkpoint, a converted .fbz) gets a sidecar `<artifact>.manifest.json`
recording exactly what produced it: scaler used, split boundaries, seed,
git commit, relevant flags. Loaders read the manifest of whatever they're
about to consume and REFUSE TO RUN if it doesn't match what they expect —
this is the regression test for the chb06 dual-scaler class of bug (§5).

Convention: manifest path is always `<artifact_path>.manifest.json`.
"""
from __future__ import annotations

import json
import os
import sys

from src.config import current_git_commit


def manifest_path(artifact_path: str) -> str:
    return artifact_path + '.manifest.json'


def write_manifest(artifact_path: str, **fields) -> str:
    """
    Write `<artifact_path>.manifest.json`. Always stamps git_commit
    automatically — every other field is the caller's responsibility
    (provenance over convenience applies here too: no silent defaults).
    """
    path = manifest_path(artifact_path)
    payload = dict(fields)
    payload.setdefault('git_commit', current_git_commit())
    payload['artifact'] = artifact_path
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path


def load_manifest(artifact_path: str, required: bool = True) -> dict | None:
    """
    Load `<artifact_path>.manifest.json`. If `required` and missing,
    hard-exit. If not required and missing, return None — the caller
    decides whether "no provenance available" is tolerable (it is, for
    artifacts that predate Gate 1b; it won't be once everything is
    regenerated under it).
    """
    path = manifest_path(artifact_path)
    if not os.path.exists(path):
        if required:
            sys.exit(
                f"ERROR: no manifest for {artifact_path} (expected {path}).\n"
                "This artifact predates Gate 1b, or was produced by a script "
                "that doesn't write manifests yet. Regenerate it with the "
                "current scripts."
            )
        return None
    with open(path) as f:
        return json.load(f)


def require_scaler_match(expected: dict, actual: dict, context: str = "") -> None:
    """
    Refuse to proceed if two scaler descriptions don't match. `expected`/
    `actual` are dicts with at minimum 'scale' and 'shift' keys (floats).
    This is the Gate 1b regression test for the chb06 class of bug: a model
    trained under scaler A evaluated against data scaled under scaler B
    should never silently produce a number.
    """
    tol = 1e-3
    scale_ok = abs(expected['scale'] - actual['scale']) < tol
    shift_ok = abs(expected['shift'] - actual['shift']) < tol
    if not (scale_ok and shift_ok):
        sys.exit(
            f"ERROR: scaler mismatch{(' — ' + context) if context else ''}.\n"
            f"  Expected (model's training manifest): "
            f"scale={expected['scale']:.4f}, shift={expected['shift']:.4f}\n"
            f"  Actual   (scaler applied this run)   : "
            f"scale={actual['scale']:.4f}, shift={actual['shift']:.4f}\n"
            "This is the chb06 dual-scaler failure mode (§5) — refusing "
            "rather than reporting a silently wrong number. Fix "
            "--scaler-source so it matches the model's training scaler."
        )
