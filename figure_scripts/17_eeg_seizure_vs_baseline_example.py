"""
Combines an existing eeg_{patient}_baseline.png and eeg_{patient}_seizure.png
into a single side-by-side comparison figure.

Fix vs. the previous version: no hardcoded "(chb10)" label, and no duplicate
caption bar squished on top of each panel's own title (that was the source of
the overlapping-text look you flagged). The patient tag is now detected
directly from the input filenames, and each source image's own embedded title
is left alone -- we just place them side by side with a gap.

Usage:
    python3 17_eeg_seizure_vs_baseline_example.py eeg_chb13_baseline.png eeg_chb13_seizure.png
    python3 17_eeg_seizure_vs_baseline_example.py eeg_chb10_baseline.png eeg_chb10_seizure.png --out custom_name.png
"""
import argparse
import re
from pathlib import Path
from PIL import Image


def detect_patient(*paths):
    for p in paths:
        m = re.search(r"(chb\d+)", Path(p).stem)
        if m:
            return m.group(1)
    return "unknown"


def combine(baseline_path, seizure_path, out_path=None, gap=20):
    baseline = Image.open(baseline_path).convert("RGB")
    seizure = Image.open(seizure_path).convert("RGB")

    if baseline.size != seizure.size:
        seizure = seizure.resize(baseline.size)

    w, h = baseline.size
    canvas = Image.new("RGB", (w * 2 + gap, h), "white")
    canvas.paste(baseline, (0, 0))
    canvas.paste(seizure, (w + gap, 0))

    patient = detect_patient(baseline_path, seizure_path)
    if out_path is None:
        out_path = f"eeg_seizure_vs_baseline_example_{patient}.png"

    canvas.save(out_path)
    print(f"Patient detected: {patient}")
    print(f"Saved -> {out_path} {canvas.size}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_path")
    parser.add_argument("seizure_path")
    parser.add_argument("--out", default=None)
    parser.add_argument("--gap", type=int, default=20)
    args = parser.parse_args()

    combine(args.baseline_path, args.seizure_path, args.out, args.gap)
