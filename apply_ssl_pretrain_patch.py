#!/usr/bin/env python3
"""
apply_ssl_pretrain_patch.py
==============================
Candidate C-i (Handoff_post_dann_scoping_to_implementation.md sec5).
Adds two functions to akida_cnn_v2.py:

  - build_seizure_cnn_v2_ssl_pretrain() -- trunk IDENTICAL (same layer
    names) to build_seizure_cnn_v2, feeding a lightweight decoder head
    that reconstructs a masked time-span from the trunk's bottleneck
    features. Decoder is training-only, discarded before the supervised
    phase -- same "throwaway auxiliary head" pattern as DANN's domain
    head, CORAL's flatten probe.
  - extract_pretrained_trunk() -- copies ONLY the trunk layers (rescaling
    through flatten) by name into a plain build_seizure_cnn_v2 instance;
    the Dense head is deliberately left at ITS OWN random init, since
    C-i's pretext task only pretrains the trunk. Mirrors extract_
    deployable_submodel()'s by-name-copy pattern but with a smaller
    layer set.

Run from ~/dissertation/ with akida_env activated:
    python3 apply_ssl_pretrain_patch.py

Hard-refuses if the anchor isn't found exactly once.
"""
import sys

PATH = 'src/models/akida_cnn_v2.py'

old = """def build_patient_adapted_model(base_model, freeze_until='relu3'):"""

new = '''def build_seizure_cnn_v2_ssl_pretrain(n_channels=18, window_samples=512, mask_len=128):
    """
    Candidate C-i pretext-task model (Handoff_post_dann_scoping_to_
    implementation.md sec5c). Trunk (rescaling -> flatten) is IDENTICAL
    (same layer names) to build_seizure_cnn_v2 -- extract_pretrained_
    trunk() below copies these weights with zero remapping. Decoder head
    is a plain 2-layer MLP reconstructing the masked time-span (n_channels
    x mask_len, flattened then reshaped) from the 1536-dim bottleneck --
    training-only, discarded entirely before the supervised phase.

    Input is the MASKED window (masked span already zeroed by the caller,
    see pretrain_ssl.py) -- this model only ever sees masked input, never
    the clean window, by construction.
    """
    inp = keras.Input(shape=(n_channels, window_samples, 1), name='eeg_input')
    x = keras.layers.Rescaling(scale=1.0 / 255.0, name='rescaling')(inp)

    x = keras.layers.Conv2D(32, (9, 7), strides=(1, 4),
        padding='valid', use_bias=False, name='conv1')(x)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.MaxPooling2D((1, 2), padding='valid', name='pool1')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu1')(x)

    x = keras.layers.Conv2D(64, (3, 3),
        padding='same', use_bias=False, name='conv2')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu2')(x)

    x = keras.layers.Conv2D(32, (3, 3),
        padding='same', use_bias=False, name='conv3')(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu3')(x)

    shared = keras.layers.Flatten(name='flatten')(x)

    # -- Decoder head (training-only, discarded before the supervised
    # phase -- reconstructs the masked [n_channels, mask_len] patch) --
    d = keras.layers.Dense(256, activation='relu', name='ssl_decoder_dense1')(shared)
    d = keras.layers.Dense(n_channels * mask_len, activation='linear',
                            name='ssl_decoder_out')(d)
    recon = keras.layers.Reshape((n_channels, mask_len), name='ssl_reconstruction')(d)

    return keras.Model(inputs=inp, outputs=recon, name='seizure_cnn_v2_ssl_pretrain')


def extract_pretrained_trunk(ssl_model, n_channels=18, window_samples=512):
    """
    Copy ONLY the trunk layers (rescaling through flatten) by name from a
    trained C-i pretext model into a plain build_seizure_cnn_v2 instance.
    The Dense head (dense1/relu_dense/output) is deliberately left at
    ITS OWN fresh random init -- C-i's pretext task pretrains the
    representation extractor, not the classifier; the supervised phase
    trains the head from scratch on top of the pretrained trunk.
    """
    target = build_seizure_cnn_v2(n_channels=n_channels, window_samples=window_samples)
    TRUNK_LAYERS = {'rescaling', 'conv1', 'bn1', 'pool1', 'relu1',
                     'conv2', 'bn2', 'pool2', 'relu2',
                     'conv3', 'bn3', 'pool3', 'relu3', 'flatten'}
    copied, skipped_head = [], []
    for layer in target.layers:
        if layer.name not in TRUNK_LAYERS:
            skipped_head.append(layer.name)   # expected -- head stays random
            continue
        try:
            src_layer = ssl_model.get_layer(layer.name)
        except ValueError:
            print(f"[extract_pretrained_trunk] WARNING -- no source layer "
                  f"for trunk layer '{layer.name}' -- investigate before "
                  "trusting this checkpoint.")
            continue
        layer.set_weights(src_layer.get_weights())
        copied.append(layer.name)
    print(f"[extract_pretrained_trunk] Copied (pretrained): {copied}")
    print(f"[extract_pretrained_trunk] Left at random init (head, by "
          f"design): {skipped_head}")
    return target


def build_patient_adapted_model(base_model, freeze_until='relu3'):'''

if __name__ == '__main__':
    with open(PATH, 'r') as f:
        content = f.read()
    n = content.count(old)
    if n == 0:
        sys.exit(f"REFUSING: anchor not found in {PATH}. No changes written.")
    if n > 1:
        sys.exit(f"REFUSING: anchor matches {n} times (expected 1). No changes written.")
    content = content.replace(old, new)
    with open(PATH, 'w') as f:
        f.write(content)
    print(f"Patched: {PATH}")
    print("\nSanity check:")
    print("  python3 -c \"from src.models.akida_cnn_v2 import "
          "build_seizure_cnn_v2_ssl_pretrain, extract_pretrained_trunk; "
          "m = build_seizure_cnn_v2_ssl_pretrain(); m.summary()\"")
