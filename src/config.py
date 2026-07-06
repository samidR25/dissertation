"""
src/config.py
==============
Single source of truth for pipeline parameters (Phase 3 Session 2, Gate 1a).

Every stage script should build a PipelineConfig (via PipelineConfig.from_args
or PipelineConfig.load_json) rather than re-deriving its own defaults. This is
the fix for the root cause behind every session-1 bug: two scripts each
independently deciding "which scaler / which split / which seed" and nobody
reconciling them. One object, one set of defaults, one place to look.

Existing scripts' own argparse CLIs are NOT being replaced (wrap, don't
rewrite) — each script keeps running standalone exactly as before. This
module is the seam: PipelineConfig.from_args(args) lets a script adopt the
shared defaults/validation without forcing every CLI flag to be renamed in
one pass. Full per-script wiring happens incrementally as each gate touches
that script (Gate 2 -> train_baseline.py freeze-depth integration, Gate 4 ->
convert_to_snn.py, etc.) — this file just has to exist and be correct first.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, fields, asdict


def current_git_commit() -> str | None:
    """Shared helper — also used by the manifest system (Gate 1b)."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


@dataclass(frozen=True)
class PipelineConfig:
    # ── Identity ──────────────────────────────────────────────────────────
    patient: str                          # target patient tag, e.g. "chb10"
    model_version: int = 2

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42

    # ── Data split & balancing ───────────────────────────────────────────
    split_ratios: tuple = (0.70, 0.15, 0.15)   # chronological train/val/test
    smote: bool = True
    scaler_source: str | None = None      # None = patient's own scaler (default);
                                           # else an override path (Gate 0b pattern)

    # ── Fine-tuning (Gate 2) ─────────────────────────────────────────────
    finetune_from: str | None = None      # base checkpoint tag, e.g. "multi"
    freeze_depth: int = 2                 # 0=full fine-tune .. 3=head-only

    # ── Evaluation ───────────────────────────────────────────────────────
    threshold: float = 0.5
    gap_tolerance_s: float = 60.0
    detection_fraction: float = 0.3
    min_sustained_windows: int = 3

    # ── Conversion (Gate 4) ──────────────────────────────────────────────
    bit_widths: tuple = (4, 4)            # (weight_bits, activation_bits) -> w4a4

    def __post_init__(self):
        assert abs(sum(self.split_ratios) - 1.0) < 1e-6, \
            f"split_ratios must sum to 1.0, got {self.split_ratios}"
        assert len(self.split_ratios) == 3, \
            "split_ratios must be (train, val, test)"
        assert 0 <= self.freeze_depth <= 3, \
            f"freeze_depth must be 0..3, got {self.freeze_depth}"
        assert 0.0 < self.threshold < 1.0, \
            f"threshold must be in (0,1), got {self.threshold}"
        assert self.seed >= 0, f"seed must be >= 0, got {self.seed}"
        assert len(self.bit_widths) == 2 and all(b > 0 for b in self.bit_widths), \
            f"bit_widths must be (weight_bits, activation_bits), both > 0, got {self.bit_widths}"
        assert self.model_version >= 1, f"model_version must be >= 1, got {self.model_version}"

    # ── Construction from an existing script's argparse Namespace ───────
    @classmethod
    def from_args(cls, args, **overrides) -> "PipelineConfig":
        """
        Build a PipelineConfig from an argparse Namespace, falling back to
        this class's own defaults for any field the script's CLI doesn't
        have. `overrides` (keyword args) take precedence over both — use
        this for fields a script computes rather than parses directly.
        """
        valid_names = {f.name for f in fields(cls)}
        kwargs = {}
        for name in valid_names:
            if name in overrides:
                kwargs[name] = overrides[name]
            elif hasattr(args, name):
                kwargs[name] = getattr(args, name)
        return cls(**kwargs)

    # ── Serialization ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d['split_ratios'] = list(d['split_ratios'])
        d['bit_widths'] = list(d['bit_widths'])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        d = dict(d)
        if 'split_ratios' in d:
            d['split_ratios'] = tuple(d['split_ratios'])
        if 'bit_widths' in d:
            d['bit_widths'] = tuple(d['bit_widths'])
        valid_names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in valid_names}
        return cls(**d)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> str:
        lines = [f"PipelineConfig({self.patient}, v{self.model_version}, seed={self.seed})"]
        for k, v in self.to_dict().items():
            if k in ('patient', 'model_version', 'seed'):
                continue
            lines.append(f"  {k:22s}: {v}")
        return "\n".join(lines)


if __name__ == '__main__':
    # Smoke test, not a unit-test suite.
    cfg = PipelineConfig(patient='chb10', finetune_from='multi', seed=123)
    print(cfg.summary())
    print()
    tmp_path = '/tmp/_pipelineconfig_smoke.json'
    cfg.save_json(tmp_path)
    reloaded = PipelineConfig.load_json(tmp_path)
    assert reloaded == cfg, "round-trip save/load did not reproduce the original config"
    print(f"Round-trip save/load OK: {tmp_path}")
    print(f"git commit: {current_git_commit()}")
