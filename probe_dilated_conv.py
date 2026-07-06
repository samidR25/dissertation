"""
Probe: dilated Conv2D compatibility with AKD1000 v1.

Extends the project's existing probe1-3b / v2-kernel-probe methodology to an
untested dimension: `dilation_rate`. The 23-constraint table has no entry for
dilation because it was never tried. This probe fills that gap the same way
the kernel-height probes did: build small variants of the *actual* v2 block
structure, run each through check_model_compatibility() -> quantize() ->
convert(), record PASS/FAIL and the stage that failed.

Design rationale (why dilation is applied to WIDTH only, not height):
    v2's actual spatial dims after block 1 (input 18,512,1 -> Conv(9,7,
    stride=(1,4),valid) -> Pool(1,2)) are roughly (10, 63). Block 2's pool
    takes height 10->5; block 3's pool takes height 5->3. A dilated 3x3
    kernel with dilation=2 has an effective height-span of 5 -- already
    larger than block 3's height dim (3). Dilating the height/channel axis
    would collide with the existing spatial-grouping design (the (9,7)
    first-layer kernel already owns "which electrodes group together").
    The actual diagnosed gap (no temporal-memory mechanism) is a WIDTH
    (time) problem, not a height (channel) problem, so every variant here
    uses dilation_rate=(1, d) -- dilate time, leave channel grouping alone.

    Also note: Keras forbids dilation_rate != 1 combined with strides != 1
    on the same Conv2D. Block 1 has stride=(1,4), so dilation is only
    tested on blocks 2 and 3 (both stride=1 by default), consistent with
    where the real architecture actually has room to take it.

Run:
    python3 probe_dilated_conv.py
"""
import sys
import numpy as np
import tf_keras as keras
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import convert, check_model_compatibility, set_akida_version, AkidaVersion

N_CHANNELS = 18
WINDOW_SAMPLES = 512


def build_variant(block2_dilation=1, block3_dilation=1, name="variant"):
    """Mirrors akida_cnn_v2.py's block structure exactly, except block2/3
    convs optionally take a (1, d) dilation_rate on the time axis."""
    inp = keras.layers.Input(shape=(N_CHANNELS, WINDOW_SAMPLES, 1), name='input')
    x = keras.layers.Rescaling(1.0 / 255, name='rescale')(inp)

    # Block 1 -- unchanged from v2 (stride present, so no dilation allowed here)
    x = keras.layers.Conv2D(32, (9, 7), strides=(1, 4), padding='valid',
                             use_bias=False, name='conv1')(x)
    x = keras.layers.MaxPooling2D((1, 2), padding='valid', name='pool1')(x)
    x = keras.layers.ReLU(max_value=6, name='relu1')(x)

    # Block 2 -- candidate dilation on time axis only
    x = keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False,
                             dilation_rate=(1, block2_dilation), name='conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = keras.layers.ReLU(max_value=6, name='relu2')(x)

    # Block 3 -- candidate dilation on time axis only
    x = keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False,
                             dilation_rate=(1, block3_dilation), name='conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(x)
    x = keras.layers.ReLU(max_value=6, name='relu3')(x)

    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(64, use_bias=False, name='dense1')(x)
    x = keras.layers.ReLU(max_value=6, name='relu_dense')(x)
    out = keras.layers.Dense(2, activation='softmax', name='output')(x)

    return keras.Model(inputs=inp, outputs=out, name=name)


def effective_kernel_span(kernel_size, dilation):
    return dilation * (kernel_size - 1) + 1


def run_probe(model, name):
    """check_model_compatibility() -> quantize() -> convert(), same 3-stage
    pipeline as every prior probe in this project."""
    result = {"name": name, "stage": None, "pass": False, "error": None}

    # Stage 1: compat check on FLOAT model
    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(model)
    except Exception as e:
        result["stage"] = "compat"
        result["error"] = str(e)
        return result

    # Stage 2: quantize (once, on float model)
    try:
        qparams = QuantizationParams(
            input_weight_bits=8,
            weight_bits=4,
            activation_bits=4,
            per_tensor_activations=True,
        )
        cal_samples = np.random.rand(8, N_CHANNELS, WINDOW_SAMPLES, 1).astype('float32') * 255.0
        q_model = quantize(model, qparams=qparams, samples=cal_samples)
    except Exception as e:
        result["stage"] = "quantize"
        result["error"] = str(e)
        return result

    # Stage 3: convert
    try:
        with set_akida_version(AkidaVersion.v1):
            convert(q_model, file_path=f'/tmp/probe_{name}.fbz')
    except Exception as e:
        result["stage"] = "convert"
        result["error"] = str(e)
        return result

    result["pass"] = True
    return result


if __name__ == '__main__':
    variants = [
        # (block2_dilation, block3_dilation, label)
        (1, 1, "control_d1_d1"),          # baseline, dilation=1 everywhere -- must PASS (sanity check on the probe itself)
        (2, 1, "block2_d2"),
        (4, 1, "block2_d4"),
        (1, 2, "block3_d2"),
        (2, 2, "both_d2_d2"),
        (2, 4, "progressive_d2_d4"),
    ]

    print(f"{'Variant':<22}{'eff.span(b2)':<14}{'eff.span(b3)':<14}{'Result':<10}Stage / error")
    print("-" * 110)

    results = []
    for b2d, b3d, label in variants:
        span2 = effective_kernel_span(3, b2d)
        span3 = effective_kernel_span(3, b3d)
        try:
            model = build_variant(block2_dilation=b2d, block3_dilation=b3d, name=label)
        except Exception as e:
            print(f"{label:<22}{span2:<14}{span3:<14}{'BUILD-FAIL':<10}{e}")
            results.append({"name": label, "pass": False, "stage": "build", "error": str(e)})
            continue

        r = run_probe(model, label)
        status = "PASS" if r["pass"] else "FAIL"
        detail = "" if r["pass"] else f"{r['stage']}: {r['error'][:80]}"
        print(f"{label:<22}{span2:<14}{span3:<14}{status:<10}{detail}")
        results.append(r)

    n_pass = sum(r["pass"] for r in results)
    print("-" * 110)
    print(f"{n_pass}/{len(results)} variants passed all three stages.")
