#!/usr/bin/env python3
"""
apply_ssl_pretrain_init_patch.py
===================================
Candidate C (Handoff_post_dann_scoping_to_implementation.md sec5d): adds
--init-from-ssl to train_baseline.py -- a THIRD model-initialisation path
alongside "fresh random init" and "--finetune-from" fine-tuning. Loads a
trunk pretrained by pretrain_ssl.py (Dense head still at random init)
instead of build_fn()'s fresh random init, then proceeds through the
EXACT SAME supervised multi-patient training path C1 already uses --
this is an initialisation change only, not an architecture or training-
loop change (sec5d: "the deployed graph is untouched").

Applies three edits:
  1. --init-from-ssl flag + validation (requires --multi-patient, mutually
     exclusive with --finetune-from/--dann/--coral/--stft/--longctx --
     first screen only, same restriction pattern as A/E).
  2. patient_tag gets an '_sslpretrain' suffix when used, so results don't
     collide with the plain C1 checkpoint.
  3. The "else: fresh training" branch gains an --init-from-ssl case that
     loads the pretrained-trunk .h5 instead of calling build_fn().

Run from ~/dissertation/ with akida_env activated, AFTER
apply_ssl_pretrain_patch.py (adds the functions this imports) and AFTER
running pretrain_ssl.py to produce the checkpoint this loads:
    python3 apply_ssl_pretrain_init_patch.py

Hard-refuses if any anchor isn't found exactly once.
"""
import sys

PATH = 'src/models/train_baseline.py'

old_1 = """parser.add_argument('--coral-lambda', type=float, default=0.01,
                    help='CORAL loss weight (default 0.01; first screen '
                         'also tries 0.1 -- CORAL losses sit on a very '
                         'different scale than cross-entropy, do not reuse '
                         'DANN lambda values, sec3d)')
args = parser.parse_args()

if args.dann and not args.multi_patient:
    parser.error('--dann requires --multi-patient (domains = pool patients)')
if args.dann and (args.finetune_from or args.stft or args.longctx):
    parser.error('--dann is mutually exclusive with --finetune-from/--stft/--longctx '
                 '(first screen only)')
if args.coral and not args.multi_patient:
    parser.error('--coral requires --multi-patient (domains = pool patients)')
if args.coral and (args.finetune_from or args.stft or args.longctx):
    parser.error('--coral is mutually exclusive with --finetune-from/--stft/--longctx '
                 '(first screen only)')
if args.dann and args.coral:
    parser.error('--dann and --coral are mutually exclusive -- run separately '
                 'for direct, uncomplicated comparability against the closed '
                 'DANN table')"""

new_1 = """parser.add_argument('--coral-lambda', type=float, default=0.01,
                    help='CORAL loss weight (default 0.01; first screen '
                         'also tries 0.1 -- CORAL losses sit on a very '
                         'different scale than cross-entropy, do not reuse '
                         'DANN lambda values, sec3d)')
parser.add_argument('--init-from-ssl', default=None,
                    help='Candidate C (Handoff_post_dann_scoping_to_'
                         'implementation.md sec5d): path to a trunk '
                         'checkpoint produced by pretrain_ssl.py (Dense '
                         'head still at random init). Loaded INSTEAD OF '
                         'fresh random init, then normal multi-patient '
                         'supervised training proceeds unchanged. Requires '
                         '--multi-patient. Mutually exclusive with '
                         '--finetune-from/--dann/--coral/--stft/--longctx '
                         '(first screen only).')
args = parser.parse_args()

if args.dann and not args.multi_patient:
    parser.error('--dann requires --multi-patient (domains = pool patients)')
if args.dann and (args.finetune_from or args.stft or args.longctx):
    parser.error('--dann is mutually exclusive with --finetune-from/--stft/--longctx '
                 '(first screen only)')
if args.coral and not args.multi_patient:
    parser.error('--coral requires --multi-patient (domains = pool patients)')
if args.coral and (args.finetune_from or args.stft or args.longctx):
    parser.error('--coral is mutually exclusive with --finetune-from/--stft/--longctx '
                 '(first screen only)')
if args.dann and args.coral:
    parser.error('--dann and --coral are mutually exclusive -- run separately '
                 'for direct, uncomplicated comparability against the closed '
                 'DANN table')
if args.init_from_ssl and not args.multi_patient:
    parser.error('--init-from-ssl requires --multi-patient (sec5e: compared '
                 'against the frozen C1 checkpoint on the same six-patient set)')
if args.init_from_ssl and (args.finetune_from or args.dann or args.coral
                            or args.stft or args.longctx):
    parser.error('--init-from-ssl is mutually exclusive with --finetune-from/'
                 '--dann/--coral/--stft/--longctx (first screen only)')"""

old_2 = """if args.multi_patient and args.finetune_from:
    patient_tag = f'multi_from_{args.finetune_from}'
elif args.multi_patient:
    patient_tag = 'multi'
else:
    patient_tag = args.patient
if args.stft:
    patient_tag += '_stft'
if args.longctx:
    patient_tag += f'_longctx_w{args.window_samples}'"""

new_2 = """if args.multi_patient and args.finetune_from:
    patient_tag = f'multi_from_{args.finetune_from}'
elif args.multi_patient:
    patient_tag = 'multi'
else:
    patient_tag = args.patient
if args.stft:
    patient_tag += '_stft'
if args.longctx:
    patient_tag += f'_longctx_w{args.window_samples}'
if args.init_from_ssl:
    patient_tag += '_sslpretrain'"""

old_3 = """else:
    # ── Fresh training ─────────────────────────────────────────────────
    model = build_fn()
    print(f"\\nBuilding new model (v{args.model_version})")"""

new_3 = """elif args.init_from_ssl:
    # ── Candidate C: pretrained-trunk init (sec5d) ───────────────────────
    if not os.path.exists(args.init_from_ssl):
        sys.exit(f"ERROR: SSL-pretrained checkpoint not found: "
                 f"{args.init_from_ssl}\\nRun pretrain_ssl.py first.")
    model = keras.models.load_model(args.init_from_ssl, compile=False)
    print(f"\\nLoaded SSL-pretrained trunk init: {args.init_from_ssl}")
    print("(Candidate C, sec5d -- trunk pretrained via masked-window "
          "reconstruction on chb01/02/05 non-seizure windows only; Dense "
          "head still at random init. Normal multi-patient supervised "
          "training proceeds unchanged from here -- this is an "
          "initialisation change only, no architecture or training-loop "
          "difference from the C1 baseline it will be compared against.)")
else:
    # ── Fresh training ─────────────────────────────────────────────────
    model = build_fn()
    print(f"\\nBuilding new model (v{args.model_version})")"""


def patch_file(path, replacements):
    with open(path, 'r') as f:
        content = f.read()
    for i, (old, new) in enumerate(replacements):
        n = content.count(old)
        if n == 0:
            sys.exit(f"REFUSING: anchor #{i} not found in {path}.\n"
                      "No changes written to this file.")
        if n > 1:
            sys.exit(f"REFUSING: anchor #{i} matches {n} times in {path} "
                      "(expected exactly 1). No changes written.")
        content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Patched: {path}")


if __name__ == '__main__':
    patch_file(PATH, [(old_1, new_1), (old_2, new_2), (old_3, new_3)])
    print("\nSanity check:")
    print("  python3 -m py_compile src/models/train_baseline.py")
    print("\nFirst screen (per sec5e -- pretrain, then supervised, then "
          "convert+eval on the same six-patient set as A):")
    print("  python3 src/models/pretrain_ssl.py --seed 123")
    print("  python3 src/models/train_baseline.py --model-version 2 "
          "--multi-patient --init-from-ssl results/pretrained_trunk_ssl_v2.h5 "
          "--seed 123")
