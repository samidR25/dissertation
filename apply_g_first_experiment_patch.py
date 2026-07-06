#!/usr/bin/env python3
"""
apply_g_first_experiment_patch.py
====================================
Candidate G phase 1 (Handoff_a_e_c_closed_to_next_steps.md sec8):
physiologically-normalised classical features (relative band power +
delta/beta ratio), C1-anchored six-patient screen.

Companion to build_dataset_g.py (a NEW file, not a patch -- run separately,
see its own docstring). This script wires --g-features through the three
existing scripts that already have an analogous --stft/--longctx flag, so
G's frozen-pool training/conversion/eval goes through the exact same
Gate-0a-compliant, manifest-checked pipeline A/C were held to. No changes
to akida_cnn_v2.py -- build_seizure_cnn_v2_3ch already exists, already
AKD1000-v1-validated, already used for stft/longctx; G reuses it unchanged
(zero new Dense parameters, zero new architecture risk, same as Gate 2's
Arm B).

Applies edits to THREE files:

  1. src/models/train_baseline.py
       - --g-features flag (mutually exclusive with --stft/--longctx/
         --dann/--coral/--init-from-ssl/--finetune-from -- first screen
         only, same discipline as every other candidate). Requires
         --multi-patient (compared against the frozen C1 checkpoint on
         the same six-patient set, same as --init-from-ssl).
       - Because --g-features implies --multi-patient, execution falls
         through the EXISTING `if args.multi_patient:` branch for the
         val split -- this is the Gate-0a-correct real/pre-SMOTE val
         already used by C1/A/C, NOT --stft's disclosed-leaky val-split
         path (train_baseline.py's own comment: "[WARNING: --stft leak
         fix is OUT OF SCOPE ... do not trust these val numbers]"). G
         gets the correct path for free by requiring --multi-patient;
         flagged here so it's an explicit design choice, not an accident.
       - data_path override to {tag}_dataset_g.npz, build_fn ->
         build_seizure_cnn_v2_3ch (identical reuse to --stft/--longctx),
         patient_tag suffix '_g'.
       - _write_ckpt_manifest()'s scaler-path resolution extended with a
         '_g' case -- WITHOUT this, a 'multi_g' patient_tag would fall
         through to the generic else-branch and resolve to the WRONG
         path (data/processed/multi_g_scaler.json, which doesn't exist --
         build_dataset_g.py writes data/processed/multi_scaler_g.json).
         Caught by tracing the manifest function before running, not
         after a wasted training run -- same class of silent-mismatch
         bug (chb10 dual-scaler, sign-flip) this project has been bitten
         by before.

  2. src/models/convert_to_snn.py
       - --g-features flag (mutually exclusive with --longctx).
       - Calibration data_path override, X_train channel handling,
         calibration shape assertion, Gate 1b per-channel-scaler skip
         note, and the cross-patient eval-data-path override (needed
         separately from the calibration path -- --eval-patient chb03
         must resolve chb03_dataset_g.npz, not chb03_dataset_ann.npz).

  3. src/evaluation/eval_event_level.py
       - --g-features flag (mutually exclusive with --longctx).
       - data_path / own_scaler_path overrides, X_eval channel handling,
         Gate 1b per-channel-scaler skip note, and the --conformal
         calibration-data channel handling (not needed for phase 1, but
         cheap to cover now so a later --conformal run against a
         personalised G checkpoint -- Phase 2, if it happens -- doesn't
         silently break on this exact same 1ch-vs-3ch assumption).

Run from ~/dissertation/ with akida_env activated:
    python3 apply_g_first_experiment_patch.py

Hard-refuses (exits nonzero, writes nothing to a given file) if any anchor
text isn't found exactly once in that file -- re-run after checking
whether the file changed since this patch was written against the
2 July 2026 snapshot (Handoff_a_e_c_closed_to_next_steps.md).
"""
import sys

TRAIN_PATH = 'src/models/train_baseline.py'
CONVERT_PATH = 'src/models/convert_to_snn.py'
EVAL_PATH = 'src/evaluation/eval_event_level.py'


def patch_file(path, replacements):
    with open(path, 'r') as f:
        content = f.read()
    for i, (old, new) in enumerate(replacements):
        n = content.count(old)
        if n == 0:
            sys.exit(f"REFUSING: anchor #{i} not found in {path}.\n"
                      "File on disk doesn't match what this patch expects "
                      "-- no changes written to this file.")
        if n > 1:
            sys.exit(f"REFUSING: anchor #{i} matches {n} times in {path} "
                      "(expected exactly 1). No changes written.")
        content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Patched: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. src/models/train_baseline.py
# ═══════════════════════════════════════════════════════════════════════════
train_edits = []

# ── 1a. Flag definition ──────────────────────────────────────────────────
old = """parser.add_argument('--longctx',          action='store_true',
                    help='Use 3-channel long-context dataset (Gate 2 — '
                         'rolling line-length + rolling delta/beta ratio). '
                         'Mutually exclusive with --stft.')"""
new = """parser.add_argument('--longctx',          action='store_true',
                    help='Use 3-channel long-context dataset (Gate 2 — '
                         'rolling line-length + rolling delta/beta ratio). '
                         'Mutually exclusive with --stft.')
parser.add_argument('--g-features',       action='store_true',
                    help='Candidate G phase 1 (Handoff_a_e_c_closed_to_'
                         'next_steps.md sec8): 3-channel relative-band-'
                         'power dataset (relative delta power, relative '
                         'beta power, delta/beta ratio) -- REPLACES raw '
                         'EEG entirely, does not augment it. Requires a '
                         'dataset built via build_dataset_g.py and '
                         '--multi-patient (compared against the frozen C1 '
                         'checkpoint on the same six-patient set, same as '
                         '--init-from-ssl). Mutually exclusive with '
                         '--stft/--longctx/--dann/--coral/--init-from-ssl/'
                         '--finetune-from (first screen only).')"""
train_edits.append((old, new))

# ── 1b. Mutual-exclusivity validation ────────────────────────────────────
old = """if args.init_from_ssl and not args.multi_patient:
    parser.error('--init-from-ssl requires --multi-patient (sec5e: compared '
                 'against the frozen C1 checkpoint on the same six-patient set)')
if args.init_from_ssl and (args.finetune_from or args.dann or args.coral
                            or args.stft or args.longctx):
    parser.error('--init-from-ssl is mutually exclusive with --finetune-from/'
                 '--dann/--coral/--stft/--longctx (first screen only)')"""
new = """if args.init_from_ssl and not args.multi_patient:
    parser.error('--init-from-ssl requires --multi-patient (sec5e: compared '
                 'against the frozen C1 checkpoint on the same six-patient set)')
if args.init_from_ssl and (args.finetune_from or args.dann or args.coral
                            or args.stft or args.longctx):
    parser.error('--init-from-ssl is mutually exclusive with --finetune-from/'
                 '--dann/--coral/--stft/--longctx (first screen only)')
if args.g_features and not args.multi_patient:
    parser.error('--g-features requires --multi-patient (sec8 phase 1: '
                 'compared against the frozen C1 checkpoint on the same '
                 'six-patient set, same requirement as --init-from-ssl)')
if args.g_features and (args.finetune_from or args.dann or args.coral
                         or args.stft or args.longctx or args.init_from_ssl):
    parser.error('--g-features is mutually exclusive with --finetune-from/'
                 '--dann/--coral/--stft/--longctx/--init-from-ssl (first '
                 'screen only)')"""
train_edits.append((old, new))

# ── 1c. patient_tag suffix ───────────────────────────────────────────────
old = """if args.stft:
    patient_tag += '_stft'
if args.longctx:
    patient_tag += f'_longctx_w{args.window_samples}'
if args.init_from_ssl:
    patient_tag += '_sslpretrain'"""
new = """if args.stft:
    patient_tag += '_stft'
if args.longctx:
    patient_tag += f'_longctx_w{args.window_samples}'
if args.g_features:
    patient_tag += '_g'
if args.init_from_ssl:
    patient_tag += '_sslpretrain'"""
train_edits.append((old, new))

# ── 1d. data_path override ───────────────────────────────────────────────
old = """if args.longctx:
    longctx_path = data_path.replace(
        '_dataset_ann.npz', f'_dataset_longctx_w{args.window_samples}.npz')
    if not os.path.exists(longctx_path):
        sys.exit(
            f"ERROR: {longctx_path} not found.\\n"
            f"Run first:\\n"
            f"  python3 src/preprocessing/preprocess.py --patient {args.patient} "
            f"--longctx --window-s <2.0 or 3.0> --longctx-lookback-s 12\\n"
            f"  python3 src/preprocessing/build_dataset_longctx.py "
            f"--patient {args.patient} --window-samples {args.window_samples}"
        )
    data_path = longctx_path

if not os.path.exists(data_path):"""
new = """if args.longctx:
    longctx_path = data_path.replace(
        '_dataset_ann.npz', f'_dataset_longctx_w{args.window_samples}.npz')
    if not os.path.exists(longctx_path):
        sys.exit(
            f"ERROR: {longctx_path} not found.\\n"
            f"Run first:\\n"
            f"  python3 src/preprocessing/preprocess.py --patient {args.patient} "
            f"--longctx --window-s <2.0 or 3.0> --longctx-lookback-s 12\\n"
            f"  python3 src/preprocessing/build_dataset_longctx.py "
            f"--patient {args.patient} --window-samples {args.window_samples}"
        )
    data_path = longctx_path

if args.g_features:
    g_path = data_path.replace('_dataset_ann.npz', '_dataset_g.npz')
    if not os.path.exists(g_path):
        sys.exit(f"ERROR: {g_path} not found.\\n"
                 f"Run: python3 src/preprocessing/build_dataset_g.py --multi-patient")
    data_path = g_path

if not os.path.exists(data_path):"""
train_edits.append((old, new))

# ── 1e. X_train/X_val channel handling ───────────────────────────────────
old = """data    = np.load(data_path)
if args.stft or args.longctx:
    X_train = data['X_train'].astype('float32')   # already (N, 18, ws, 3)
    X_val   = data['X_val'].astype('float32')
else:
    X_train = data['X_train'][..., np.newaxis].astype('float32')
    X_val   = data['X_val']  [..., np.newaxis].astype('float32')"""
new = """data    = np.load(data_path)
if args.stft or args.longctx or args.g_features:
    X_train = data['X_train'].astype('float32')   # already (N, 18, ws, 3)
    X_val   = data['X_val'].astype('float32')
else:
    X_train = data['X_train'][..., np.newaxis].astype('float32')
    X_val   = data['X_val']  [..., np.newaxis].astype('float32')"""
train_edits.append((old, new))

# ── 1f. X_test channel handling ──────────────────────────────────────────
old = """if has_test:
    X_test = data['X_test'].astype('float32') if (args.stft or args.longctx) \\
             else data['X_test'][..., np.newaxis].astype('float32')
    y_test = data['y_test']"""
new = """if has_test:
    X_test = data['X_test'].astype('float32') if (args.stft or args.longctx or args.g_features) \\
             else data['X_test'][..., np.newaxis].astype('float32')
    y_test = data['y_test']"""
train_edits.append((old, new))

# ── 1g. expected_shape check ─────────────────────────────────────────────
old = """if args.longctx:
    expected_shape = (18, args.window_samples, 3)
elif args.stft:
    expected_shape = (18, 512, 3)
else:
    expected_shape = (18, 512, 1)"""
new = """if args.longctx:
    expected_shape = (18, args.window_samples, 3)
elif args.stft or args.g_features:
    expected_shape = (18, 512, 3)
else:
    expected_shape = (18, 512, 1)"""
train_edits.append((old, new))

# ── 1h. build_fn selection ───────────────────────────────────────────────
old = """if args.stft:
    from src.models.akida_cnn_v2 import build_seizure_cnn_v2_3ch
    build_fn = lambda: build_seizure_cnn_v2_3ch(n_channels=18, window_samples=512)
elif args.longctx:"""
new = """if args.stft or args.g_features:
    from src.models.akida_cnn_v2 import build_seizure_cnn_v2_3ch
    build_fn = lambda: build_seizure_cnn_v2_3ch(n_channels=18, window_samples=512)
elif args.longctx:"""
train_edits.append((old, new))

# ── 1i. _write_ckpt_manifest scaler-path resolution ──────────────────────
old = """    if '_longctx_w' in patient_tag:
        _base_tag, _ws = patient_tag.split('_longctx_w')
        scaler_path = (f'data/processed/multi_scaler_longctx_w{_ws}.json'
                       if _base_tag == 'multi' or _base_tag.startswith('multi_from_')
                       or _base_tag.startswith('multi_sslpretrain')
                       else f'data/processed/{_base_tag}_scaler_longctx_w{_ws}.json')
    else:
        scaler_path = ('data/processed/multi_scaler.json'
                       if patient_tag == 'multi' or patient_tag.startswith('multi_from_')
                       or patient_tag.startswith('multi_sslpretrain')
                       else f'data/processed/{patient_tag}_scaler.json')"""
new = """    if '_longctx_w' in patient_tag:
        _base_tag, _ws = patient_tag.split('_longctx_w')
        scaler_path = (f'data/processed/multi_scaler_longctx_w{_ws}.json'
                       if _base_tag == 'multi' or _base_tag.startswith('multi_from_')
                       or _base_tag.startswith('multi_sslpretrain')
                       else f'data/processed/{_base_tag}_scaler_longctx_w{_ws}.json')
    elif patient_tag.endswith('_g'):
        # Candidate G (sec8) -- without this branch, patient_tag='multi_g'
        # would fall through to the generic else-branch below and resolve
        # to data/processed/multi_g_scaler.json, which does not exist --
        # build_dataset_g.py writes multi_scaler_g.json / {tag}_scaler_g.json.
        _base_tag = patient_tag[:-len('_g')]
        scaler_path = ('data/processed/multi_scaler_g.json'
                       if _base_tag == 'multi' or _base_tag.startswith('multi_from_')
                       or _base_tag.startswith('multi_sslpretrain')
                       else f'data/processed/{_base_tag}_scaler_g.json')
    else:
        scaler_path = ('data/processed/multi_scaler.json'
                       if patient_tag == 'multi' or patient_tag.startswith('multi_from_')
                       or patient_tag.startswith('multi_sslpretrain')
                       else f'data/processed/{patient_tag}_scaler.json')"""
train_edits.append((old, new))


# ═══════════════════════════════════════════════════════════════════════════
# 2. src/models/convert_to_snn.py
# ═══════════════════════════════════════════════════════════════════════════
convert_edits = []

# ── 2a. Flag definition ──────────────────────────────────────────────────
old = """parser.add_argument('--longctx',        action='store_true',
                    help='Use 3-channel long-context dataset for PTQ '
                         'calibration (Gate 2 Arms B/C).')
parser.add_argument('--window-samples', type=int, default=512,
                    choices=[512, 768],
                    help='Window size in samples. Only meaningful with --longctx.')
args = parser.parse_args()"""
new = """parser.add_argument('--longctx',        action='store_true',
                    help='Use 3-channel long-context dataset for PTQ '
                         'calibration (Gate 2 Arms B/C).')
parser.add_argument('--window-samples', type=int, default=512,
                    choices=[512, 768],
                    help='Window size in samples. Only meaningful with --longctx.')
parser.add_argument('--g-features',     action='store_true',
                    help='Use Candidate G relative-band-power dataset for '
                         'PTQ calibration/eval (Handoff_a_e_c_closed_to_'
                         'next_steps.md sec8 phase 1). Mutually exclusive '
                         'with --longctx. NOTE: pass --patient multi (not '
                         'multi_g) for calibration-data resolution -- use '
                         '--base multi_g to load the checkpoint, same '
                         'decoupling already used for --coral/--dann via '
                         '--variant.')
args = parser.parse_args()"""
convert_edits.append((old, new))

# ── 2b. Calibration data_path override ───────────────────────────────────
old = """if args.longctx:
    data_path = (f'data/processed/{args.patient}_dataset_longctx_w'
                 f'{args.window_samples}.npz')
else:
    data_path = f'data/processed/{args.patient}_dataset_ann.npz'
if not os.path.exists(data_path):
    sys.exit(
        f"ERROR: {data_path} not found.\\n"
        + (f"Run: python3 src/preprocessing/build_dataset_longctx.py "
           f"--patient {args.patient} --window-samples {args.window_samples}"
           if args.longctx else
           f"Run: python3 src/preprocessing/build_dataset.py --patient {args.patient}")
    )"""
new = """if args.longctx and args.g_features:
    sys.exit("ERROR: --longctx and --g-features are mutually exclusive.")
if args.longctx:
    data_path = (f'data/processed/{args.patient}_dataset_longctx_w'
                 f'{args.window_samples}.npz')
elif args.g_features:
    data_path = f'data/processed/{args.patient}_dataset_g.npz'
else:
    data_path = f'data/processed/{args.patient}_dataset_ann.npz'
if not os.path.exists(data_path):
    if args.longctx:
        _hint = (f"Run: python3 src/preprocessing/build_dataset_longctx.py "
                f"--patient {args.patient} --window-samples {args.window_samples}")
    elif args.g_features:
        _hint = (f"Run: python3 src/preprocessing/build_dataset_g.py "
                f"--patient {args.patient}  (or --multi-patient, then pass "
                f"--patient multi here)")
    else:
        _hint = f"Run: python3 src/preprocessing/build_dataset.py --patient {args.patient}"
    sys.exit(f"ERROR: {data_path} not found.\\n{_hint}")"""
convert_edits.append((old, new))

# ── 2c. Calibration X_train channel handling ─────────────────────────────
old = """data    = np.load(data_path)
if args.longctx:
    X_train = data['X_train'].astype('float32')
else:
    X_train = data['X_train'][..., np.newaxis].astype('float32')
y_train = data['y_train']"""
new = """data    = np.load(data_path)
if args.longctx or args.g_features:
    X_train = data['X_train'].astype('float32')
else:
    X_train = data['X_train'][..., np.newaxis].astype('float32')
y_train = data['y_train']"""
convert_edits.append((old, new))

# ── 2d. Calibration shape assertion ──────────────────────────────────────
old = """if args.longctx:
    assert X_cal.shape[1:] == (18, args.window_samples, 3), \\
        f"Wrong calibration shape {X_cal.shape}"
else:
    assert X_cal.shape[1:] == (18, 512, 1), \\
        f"Wrong calibration shape {X_cal.shape}"
"""
new = """if args.longctx:
    assert X_cal.shape[1:] == (18, args.window_samples, 3), \\
        f"Wrong calibration shape {X_cal.shape}"
elif args.g_features:
    assert X_cal.shape[1:] == (18, 512, 3), \\
        f"Wrong calibration shape {X_cal.shape}"
else:
    assert X_cal.shape[1:] == (18, 512, 1), \\
        f"Wrong calibration shape {X_cal.shape}"
"""
convert_edits.append((old, new))

# ── 2e. Gate 1b per-channel-scaler skip note ─────────────────────────────
old = """elif ckpt_manifest.get('scaler') and 'per_patient' not in ckpt_manifest['scaler']:
    if args.longctx:
        # Longctx scalers use per-channel {ch0_min,...} keys, not flat
        # {scale,shift}. Skip the flat-format consistency check — Gate 2c
        # in train_baseline.py already verified the correct scaler was used.
        print("  [Gate 1b] Longctx checkpoint — per-channel scaler format; "
              "flat-format consistency check skipped (Gate 2c verified at training).")
    else:"""
new = """elif ckpt_manifest.get('scaler') and 'per_patient' not in ckpt_manifest['scaler']:
    if args.longctx or args.g_features:
        # Longctx/G scalers use per-channel {ch0_min,...} keys, not flat
        # {scale,shift}. Skip the flat-format consistency check — Gate 2c
        # in train_baseline.py already verified the correct scaler was used.
        _tag = 'Longctx' if args.longctx else 'Candidate G'
        print(f"  [Gate 1b] {_tag} checkpoint — per-channel scaler format; "
              "flat-format consistency check skipped (Gate 2c verified at training).")
    else:"""
convert_edits.append((old, new))

# ── 2f. Cross-patient eval data_path override ────────────────────────────
old = """if eval_tag != args.patient:
    eval_data_path = f'data/processed/{eval_tag}_dataset_ann.npz'
    if not os.path.exists(eval_data_path):
        sys.exit(
            f"ERROR: {eval_data_path} not found.\\n"
            f"Run: python3 src/preprocessing/build_dataset.py --patient {eval_tag}"
        )
    eval_data = np.load(eval_data_path)
else:
    eval_data = data  # same data already loaded for calibration"""
new = """if eval_tag != args.patient:
    if args.g_features:
        eval_data_path = f'data/processed/{eval_tag}_dataset_g.npz'
        _eval_hint = (f"Run: python3 src/preprocessing/build_dataset_g.py "
                     f"--patient {eval_tag}")
    else:
        eval_data_path = f'data/processed/{eval_tag}_dataset_ann.npz'
        _eval_hint = (f"Run: python3 src/preprocessing/build_dataset.py "
                     f"--patient {eval_tag}")
    if not os.path.exists(eval_data_path):
        sys.exit(f"ERROR: {eval_data_path} not found.\\n{_eval_hint}")
    eval_data = np.load(eval_data_path)
else:
    eval_data = data  # same data already loaded for calibration"""
convert_edits.append((old, new))

# ── 2g. X_eval channel handling (both has_test branches) ────────────────
old = """has_test = 'X_test' in eval_data.files and eval_data['y_test'].sum() > 0
if has_test:
    if args.longctx:
        X_eval = eval_data['X_test'].astype('float32')
    else:
        X_eval = eval_data['X_test'][..., np.newaxis].astype('float32')
    y_eval = eval_data['y_test']
    eval_label = 'test'
else:
    if args.longctx:
        X_eval_train = eval_data['X_train'].astype('float32')
    else:
        X_eval_train = eval_data['X_train'][..., np.newaxis].astype('float32')
    y_eval_train = eval_data['y_train']
    X_eval = X_eval_train[:500]
    y_eval = y_eval_train[:500]
    eval_label = 'train (no test seizures)'"""
new = """has_test = 'X_test' in eval_data.files and eval_data['y_test'].sum() > 0
if has_test:
    if args.longctx or args.g_features:
        X_eval = eval_data['X_test'].astype('float32')
    else:
        X_eval = eval_data['X_test'][..., np.newaxis].astype('float32')
    y_eval = eval_data['y_test']
    eval_label = 'test'
else:
    if args.longctx or args.g_features:
        X_eval_train = eval_data['X_train'].astype('float32')
    else:
        X_eval_train = eval_data['X_train'][..., np.newaxis].astype('float32')
    y_eval_train = eval_data['y_train']
    X_eval = X_eval_train[:500]
    y_eval = y_eval_train[:500]
    eval_label = 'train (no test seizures)'"""
convert_edits.append((old, new))


# ═══════════════════════════════════════════════════════════════════════════
# 3. src/evaluation/eval_event_level.py
# ═══════════════════════════════════════════════════════════════════════════
eval_edits = []

# ── 3a. Flag definition ──────────────────────────────────────────────────
old = """parser.add_argument('--longctx',        action='store_true')
parser.add_argument('--window-samples', type=int, default=512, choices=[512, 768])"""
new = """parser.add_argument('--longctx',        action='store_true')
parser.add_argument('--window-samples', type=int, default=512, choices=[512, 768])
parser.add_argument('--g-features',     action='store_true',
                    help='Evaluate a Candidate G checkpoint (relative-band-'
                         'power 3-channel dataset, Handoff_a_e_c_closed_to_'
                         'next_steps.md sec8 phase 1). Mutually exclusive '
                         'with --longctx.')"""
eval_edits.append((old, new))

# ── 3b. data_path resolution ─────────────────────────────────────────────
old = """if args.longctx:
    data_path = (f'data/processed/{args.eval_patient}_dataset_longctx_w'
                 f'{args.window_samples}.npz')
else:
    data_path = f'data/processed/{args.eval_patient}_dataset_ann.npz'
assert os.path.exists(data_path), f"Dataset not found: {data_path}\""""
new = """if args.longctx and args.g_features:
    sys.exit("ERROR: --longctx and --g-features are mutually exclusive.")
if args.longctx:
    data_path = (f'data/processed/{args.eval_patient}_dataset_longctx_w'
                 f'{args.window_samples}.npz')
elif args.g_features:
    data_path = f'data/processed/{args.eval_patient}_dataset_g.npz'
else:
    data_path = f'data/processed/{args.eval_patient}_dataset_ann.npz'
assert os.path.exists(data_path), f"Dataset not found: {data_path}\""""
eval_edits.append((old, new))

# ── 3c. X_eval channel handling ──────────────────────────────────────────
old = """data = np.load(data_path)
assert 'X_test' in data.files, f"{data_path} has no X_test split"
if args.longctx:
    X_eval = data['X_test'].astype('float32')
else:
    X_eval = data['X_test'][..., np.newaxis].astype('float32')
y_eval = data['y_test']"""
new = """data = np.load(data_path)
assert 'X_test' in data.files, f"{data_path} has no X_test split"
if args.longctx or args.g_features:
    X_eval = data['X_test'].astype('float32')
else:
    X_eval = data['X_test'][..., np.newaxis].astype('float32')
y_eval = data['y_test']"""
eval_edits.append((old, new))

# ── 3d. own_scaler_path resolution (Gate 0b) ─────────────────────────────
old = """if args.longctx:
    own_scaler_path = (f'data/processed/{args.eval_patient}_scaler_longctx_w'
                       f'{args.window_samples}.json')
else:
    own_scaler_path = f'data/processed/{args.eval_patient}_scaler.json'"""
new = """if args.longctx:
    own_scaler_path = (f'data/processed/{args.eval_patient}_scaler_longctx_w'
                       f'{args.window_samples}.json')
elif args.g_features:
    own_scaler_path = f'data/processed/{args.eval_patient}_scaler_g.json'
else:
    own_scaler_path = f'data/processed/{args.eval_patient}_scaler.json'"""
eval_edits.append((old, new))

# ── 3e. Gate 1b longctx-specific skip message ────────────────────────────
old = """        if expected_scaler is not None:
            if args.longctx:
                print(f"\\n[Gate 1b] Longctx run — per-channel scaler; "
                      "require_scaler_match skipped (Gate 2c verified at training).")
            else:"""
new = """        if expected_scaler is not None:
            if args.longctx or args.g_features:
                _tag = 'Longctx' if args.longctx else 'Candidate G'
                print(f"\\n[Gate 1b] {_tag} run — per-channel scaler; "
                      "require_scaler_match skipped (Gate 2c verified at training).")
            else:"""
eval_edits.append((old, new))

# ── 3f. Conformal calibration X_calib channel handling ───────────────────
old = """    if args.longctx:
        X_calib = data['X_val'].astype('float32')
    else:
        X_calib = data['X_val'][..., np.newaxis].astype('float32')"""
new = """    if args.longctx or args.g_features:
        X_calib = data['X_val'].astype('float32')
    else:
        X_calib = data['X_val'][..., np.newaxis].astype('float32')"""
eval_edits.append((old, new))

# ── 3g. Guard --g-features against --scaler-source ───────────────────────
# --scaler-source assumes a flat {scale,shift} JSON (own_scaler['scale']/
# ['shift']) -- G's scaler is per-channel ({ch0_min,...}, same format as
# --longctx). This exact gap already exists for --longctx in the unpatched
# code (pre-existing, out of scope to fix here) -- but there's no reason to
# let --g-features hit the same confusing KeyError instead of a clear
# error message, so it's guarded explicitly for the new flag.
old = """if args.scaler_source:
    assert os.path.exists(args.scaler_source), f"Scaler not found: {args.scaler_source}\""""
new = """if args.scaler_source and args.g_features:
    sys.exit("ERROR: --scaler-source is not supported with --g-features -- "
             "G's scaler is per-channel format ({ch0_min,...}), not the flat "
             "{scale,shift} format --scaler-source's override logic assumes "
             "(same limitation --longctx already has, pre-existing). Not "
             "needed for phase 1 (frozen-pool eval on held-out patients, no "
             "dual-scaler correction involved).")
if args.scaler_source:
    assert os.path.exists(args.scaler_source), f"Scaler not found: {args.scaler_source}\""""
eval_edits.append((old, new))


if __name__ == '__main__':
    patch_file(TRAIN_PATH, train_edits)
    patch_file(CONVERT_PATH, convert_edits)
    patch_file(EVAL_PATH, eval_edits)
    print("\nAll patches applied successfully.")
    print("\nSanity check:")
    print("  python3 -m py_compile src/models/train_baseline.py "
          "src/models/convert_to_snn.py src/evaluation/eval_event_level.py")
    print("\nNext steps (phase 1, sec8):")
    print("  1. python3 src/preprocessing/build_dataset_g.py --multi-patient")
    print("  2. python3 src/preprocessing/build_dataset_g.py --patient chb03  "
          "# repeat for chb10 chb13 chb15 chb16 chb20")
    print("  3. python3 src/models/train_baseline.py --model-version 2 "
          "--multi-patient --g-features --seed 123")
    print("  4. python3 src/models/convert_to_snn.py --patient multi "
          "--base multi_g --eval-patient chb03 --g-features "
          "--model-version 2 --cal-samples 1024")
    print("     # repeat --eval-patient for chb10 chb13 chb15 chb16 chb20")
    print("  5. python3 src/evaluation/eval_event_level.py --fbz "
          "results/seizure_model_multi_g_v2_w4a4.fbz --eval-patient chb03 "
          "--g-features  # repeat per patient")
