"""
generate_pipeline_diagram.py

Produces a "Study Overview"-style pipeline diagram (boxed panels, connected
by red arrows, high-contrast black borders on white) matching the visual
style of the reference image, but built around this project's ACTUAL
pipeline stages rather than the reference's own (which used a different
project's methodology -- 10-fold cross-val, dB/Mel-filter feature
extraction, generic "Recognition Models" -- none of which apply here).

Run with: python3 generate_pipeline_diagram.py
Requires only matplotlib (no other dependencies).
Output: pipeline_architecture_full.png in the current directory, at 300 DPI
so text stays legible even when placed in a rotated/sideways LaTeX figure.

TO INSERT A REAL HARDWARE PHOTO (Raspberry Pi 5 + AKD1000 + HAT+):
see the clearly marked HARDWARE_PHOTO_PATH variable near the top --
set it to your photo's file path and re-run. If left as None, a
placeholder box is drawn instead so the diagram still renders.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────
HARDWARE_PHOTO_PATH = None #"hardware_setup.jpeg"
OUTPUT_PATH = "pipeline_architecture_full.png"
DPI = 300

BOX_EDGE = "black"
BOX_FACE = "white"
ARROW_COLOR = "#d62728"  # matches the reference image's red arrows
GROUP_LW = 2.2
BOX_LW = 1.4
FONT_TITLE = 10
FONT_LABEL = 8.5

fig, ax = plt.subplots(figsize=(20, 7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 38)
ax.axis("off")


def group_box(x, y, w, h, label=None, label_y_offset=1.5):
    """Outer bounding box for a pipeline stage group (like the reference
    image's 4 big black-bordered rectangles)."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.6",
        linewidth=GROUP_LW, edgecolor=BOX_EDGE, facecolor="none"
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + w / 2, y + h + label_y_offset, label,
                 ha="center", va="bottom", fontsize=FONT_TITLE + 1, fontweight="bold")
    return rect


def inner_box(x, y, w, h, title, subtitle=None, fontsize=FONT_LABEL):
    """A single stage box inside a group, matching the reference's
    white-filled, black-bordered inner boxes."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.3",
        linewidth=BOX_LW, edgecolor=BOX_EDGE, facecolor=BOX_FACE
    )
    ax.add_patch(rect)
    ty = y + h * 0.62 if subtitle else y + h * 0.5
    ax.text(x + w / 2, ty, title, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", wrap=True)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.22, subtitle, ha="center", va="center",
                fontsize=fontsize - 1, style="italic")
    return rect


def big_arrow(x0, y0, x1, y1):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=28,
        linewidth=3.2, color=ARROW_COLOR
    )
    ax.add_patch(arr)


def small_waveform(x, y, w, h, seed=0, seizure=False):
    """Tiny inline EEG-like waveform sketch (no data dependency -- purely
    illustrative, matching the reference image's icon style)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 120)
    if seizure:
        sig = 0.5 * np.sin(2 * np.pi * 9 * t) + 0.15 * rng.standard_normal(120)
    else:
        sig = 0.15 * np.sin(2 * np.pi * 2 * t) + 0.08 * rng.standard_normal(120)
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    xs = x + t * w
    ys = y + h / 2 + sig * (h * 0.35)
    ax.plot(xs, ys, color="#1f77b4" if not seizure else "#d62728", linewidth=1.1)


# ══════════════════════════════════════════════════════════════════════
# GROUP 1 — Data ingestion & preparation (mirrors §3.1)
# ══════════════════════════════════════════════════════════════════════
g1 = group_box(1, 4, 20, 30, label="1. Dataset & Chronological Split")

inner_box(2.5, 27, 17, 5.5, "Select EEG\n(CHB-MIT, 18ch)")
small_waveform(3.5, 24, 15, 2.2, seed=1)

big_arrow(11, 24, 11, 21.5)

inner_box(2.5, 15.5, 17, 5.5, "Chronological Split", "70 : 15 : 15 (train:val:test)")

big_arrow(11, 15.5, 11, 13)

inner_box(2.5, 6, 17, 5.5, "Label Segments", "seizure / non-seizure windows,\npost-split SMOTE only")

big_arrow(21, 19, 25, 19)

# ══════════════════════════════════════════════════════════════════════
# GROUP 2 — Model input representation (mirrors §3.2, primary vs STFT variant)
# ══════════════════════════════════════════════════════════════════════
g2 = group_box(25, 4, 20, 30, label="2. Input Representation")

inner_box(26.5, 27, 17, 5.5, "Raw Single-Channel\n(primary path)")
big_arrow(35, 27, 35, 22.5)
inner_box(26.5, 16.5, 17, 4.5, "STFT, 3-Channel", "(secondary variant, evaluated\nbut not the reported default)")
big_arrow(35, 16.5, 35, 12.5)
inner_box(26.5, 6, 17, 5, "seizure\\_cnn\\_v2 Input", "1-channel raw waveform\n(unless STFT variant selected)")

big_arrow(45, 19, 49, 19)

# ══════════════════════════════════════════════════════════════════════
# GROUP 3 — Recognition model + training regimes (mirrors §3.2-§3.3)
# ══════════════════════════════════════════════════════════════════════
g3 = group_box(49, 4, 24, 30, label="3. Model & Training Regime")

inner_box(50.5, 27, 21, 5.5, "seizure\\_cnn\\_v2", "3x (Conv-BN-Pool-ReLU)\n-> Flatten -> Dense(64) -> Dense(2)")
big_arrow(61, 27, 61, 23)
inner_box(50.5, 17.5, 21, 5, "Training Regime", "C1 (frozen) / pool7 & pool6-X\n/ C2 (per-patient fine-tune)")
big_arrow(61, 17.5, 61, 13.5)
inner_box(50.5, 6, 21, 5.5, "Evaluation", "LOPO (full-recording) or\nchronological test-slice")

big_arrow(73, 19, 77, 19)

# ══════════════════════════════════════════════════════════════════════
# GROUP 4 — Quantisation, conversion, hardware deployment (mirrors §3.2.2-3.5)
# ══════════════════════════════════════════════════════════════════════
g4 = group_box(77, 4, 21, 30, label="4. Deploy to AKD1000")

inner_box(78.5, 27, 18, 5, "Quantise (w4a4)", "quantizeml")
big_arrow(87.5, 27, 87.5, 23.5)
inner_box(78.5, 18, 18, 4.5, "Convert to SNN", "cnn2snn, AkidaVersion.v1")
big_arrow(87.5, 18, 87.5, 14.5)

HW_X, HW_Y, HW_W, HW_H = 78.5, 6.5, 18, 6  # box position/size, data coords

if HARDWARE_PHOTO_PATH:
    img = plt.imread(HARDWARE_PHOTO_PATH)
    # Draw the bounding box first (so the photo has a visible border matching
    # every other box in the diagram), then place the image to EXACTLY fill
    # that box's data-coordinate extent -- this guarantees correct sizing
    # regardless of the photo's actual pixel resolution, unlike OffsetImage's
    # zoom parameter (which scales relative to raw pixel size and will blow
    # up or shrink unpredictably depending on the source photo's resolution).
    rect = FancyBboxPatch(
        (HW_X, HW_Y), HW_W, HW_H,
        boxstyle="round,pad=0.15,rounding_size=0.3",
        linewidth=BOX_LW, edgecolor=BOX_EDGE, facecolor="none", zorder=2
    )
    ax.add_patch(rect)
    ax.imshow(
        img,
        extent=[HW_X + 0.3, HW_X + HW_W - 0.3, HW_Y + 0.3, HW_Y + HW_H - 0.3],
        aspect="auto",  # fills the box exactly; use "equal" instead if you
                         # want the photo's true aspect ratio preserved with
                         # letterboxing rather than stretched-to-fit
        zorder=1
    )
else:
    inner_box(HW_X, HW_Y, HW_W, HW_H, " Pi 5 + AKD1000 + HAT + Pi Fan (soldered to GPIO pins)")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")
