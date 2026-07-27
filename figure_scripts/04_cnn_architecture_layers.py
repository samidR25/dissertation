"""
Generates cnn_architecture_layers.png by DYNAMICALLY INTROSPECTING your real
seizure_cnn_v2 model -- no hardcoded parameter counts. It imports
build_seizure_cnn_v2 straight from src/models/akida_cnn_v2.py in your repo
(matching the same import style your other scripts use), builds it, and reads
each layer's actual output shape and parameter count via Keras itself.

Run from your repo root (~/dissertation), inside akida_env:

    conda activate akida_env
    cd ~/dissertation
    python3 /path/to/04_cnn_architecture_layers.py

If you're running it from somewhere else, point it at the repo root explicitly:

    python3 04_cnn_architecture_layers.py --repo-root ~/dissertation

Optional override if your trained checkpoints used a different window size:
    python3 04_cnn_architecture_layers.py --window-samples 256
"""
import argparse
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


def introspect_model(n_channels: int, window_samples: int, repo_root: str):
    sys.path.insert(0, repo_root)
    try:
        from src.models.akida_cnn_v2 import build_seizure_cnn_v2  # your real repo code
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"\nCould not import src.models.akida_cnn_v2 from repo_root='{repo_root}'.\n"
            f"Original error: {e}\n\n"
            "This script expects the same layout your other scripts use "
            "(sys.path.insert(0, '.') + 'from src.models.akida_cnn_v2 import ...'),\n"
            "i.e. it needs to see a 'src/models/akida_cnn_v2.py' under repo_root.\n"
            "Fix: run this script with --repo-root pointing at ~/dissertation, e.g.\n"
            "  python3 04_cnn_architecture_layers.py --repo-root ~/dissertation\n"
        )

    model = build_seizure_cnn_v2(n_channels=n_channels, window_samples=window_samples)

    layers_info = []
    for layer in model.layers:
        cfg = {
            "name": layer.name,
            "class": layer.__class__.__name__,
            "output_shape": layer.output_shape if hasattr(layer, "output_shape") else None,
            "params": int(layer.count_params()),
        }
        layers_info.append(cfg)

    total_params = int(model.count_params())
    return layers_info, total_params


def group_into_blocks(layers_info):
    """Collapse Conv->BN->Pool->ReLU runs into single display blocks, and
    sum real params for the trunk (through relu3) vs the dense head."""
    blocks = []
    trunk_params = 0
    head_params = 0

    i = 0
    while i < len(layers_info):
        li = layers_info[i]
        cname = li["class"]

        if cname == "InputLayer":
            i += 1
            continue
        if cname == "Rescaling":
            blocks.append(("Input\n(rescaled)", "#B0B0B0", 0))
            i += 1
            continue
        if cname == "Conv2D":
            group = [li]
            j = i + 1
            while j < len(layers_info) and layers_info[j]["class"] in (
                "BatchNormalization", "MaxPooling2D", "ReLU"
            ):
                group.append(layers_info[j])
                j += 1
            block_params = sum(g["params"] for g in group)
            trunk_params += block_params
            label = f"Conv block\n({li['name']})"
            blocks.append((label, "#4C72B0", block_params))
            i = j
            continue
        if cname == "Flatten":
            blocks.append(("Flatten", "#C0A0D0", 0))
            i += 1
            continue
        if cname == "Dense":
            group = [li]
            j = i + 1
            while j < len(layers_info) and layers_info[j]["class"] == "ReLU":
                group.append(layers_info[j])
                j += 1
            block_params = sum(g["params"] for g in group)
            head_params += block_params
            label = f"Dense({li['output_shape'][-1]})\n({li['name']})"
            color = "#DD8452" if li["output_shape"][-1] != 2 else "#C44E52"
            blocks.append((label, color, block_params))
            i = j
            continue
        i += 1

    return blocks, trunk_params, head_params


def draw_diagram(blocks, trunk_params, head_params, total_params, out_path):
    n = len(blocks)
    box_w, box_h = 1.9, 1.5
    gap = 0.6
    xs = [i * (box_w + gap) for i in range(n)]

    fig_w = max(14, xs[-1] + box_w + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))

    y0 = 0
    for i, (label, color, params) in enumerate(blocks):
        x = xs[i]
        rect = mpatches.FancyBboxPatch(
            (x, y0 - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.06", linewidth=1.1,
            edgecolor='black', facecolor=color, alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y0 + 0.18, label, ha='center', va='center',
                fontsize=8.6, color='white', weight='bold')
        if params:
            ax.text(x + box_w / 2, y0 - 0.32, f"{params:,} params",
                    ha='center', va='center', fontsize=7.6, color='white')
        if i < n - 1:
            arrow = FancyArrowPatch(
                (x + box_w, y0), (xs[i + 1], y0),
                arrowstyle='-|>', mutation_scale=12, color='black', linewidth=1.2,
            )
            ax.add_patch(arrow)

    ax.text(
        xs[-1] / 2, -1.9,
        f"Convolutional trunk (through relu3): {trunk_params:,} real parameters   |   "
        f"Dense head (flatten\u2192output): {head_params:,} real parameters   |   "
        f"Total: {total_params:,} real parameters",
        ha='center', va='center', fontsize=9.5, style='italic',
    )

    ax.set_xlim(-1, xs[-1] + box_w + 1)
    ax.set_ylim(-2.6, 1.6)
    ax.axis('off')
    ax.set_title(
        'seizure_cnn_v2: layer-by-layer architecture\n'
        '(introspected live from build_seizure_cnn_v2() -- not hardcoded)',
        fontsize=12.5, pad=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")
    print(f"Trunk params: {trunk_params:,} | Head params: {head_params:,} | Total: {total_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-channels", type=int, default=18)
    parser.add_argument("--window-samples", type=int, default=512)
    parser.add_argument("--out", default="cnn_architecture_layers.png")
    parser.add_argument("--repo-root", default=".",
                         help="Path to your dissertation repo root (the directory "
                              "containing src/models/akida_cnn_v2.py). Default '.' "
                              "assumes you're running this from ~/dissertation.")
    args = parser.parse_args()

    layers_info, total_params = introspect_model(args.n_channels, args.window_samples, args.repo_root)
    blocks, trunk_params, head_params = group_into_blocks(layers_info)
    draw_diagram(blocks, trunk_params, head_params, total_params, args.out)
