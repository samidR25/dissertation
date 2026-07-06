#!/usr/bin/env python3
"""
apply_force_scaler_mismatch_patch.py
============================================
C2 compounding, action item 2 (Handoff_g_closed_to_c2_compounding.md sec4,
Handoff_item1_done_item2_next.md sec2): chb15 amplitude-calibration +
C2 stacking.

Problem found this session, not anticipated in the original scoping: Gate 1b
(chb06 dual-scaler fix, Methodology_Ledger_Consolidated.md sec5) hard-refuses
any --scaler-source override whose scale/shift doesn't match the model's own
recorded training scaler. Every chb15 checkpoint -- C1 and every ft seed --
has chb15's own scaler baked into its manifest, so stacking the amplitude-
calibration scaler on top of the personalised checkpoint at eval time
trips Gate 1b every time, by design. This is the gate working correctly
(catching a real scale mismatch), not a bug -- but this project's amplitude-
calibration technique itself (sec3b) IS a deliberate, controlled scale
shift, done for a legitimate reason. Gate 1b needs to distinguish
"accidental mismatch" (refuse) from "deliberate, disclosed override for a
controlled experiment" (allow, but log loudly) -- it currently can't.

Applies one narrow capability, not a bypass of the check's substance:

  1. src/evaluation/eval_event_level.py
       - New --force-scaler-mismatch flag (default off, unchanged behaviour
         unless explicitly passed). When set AND --scaler-source is also
         set, Gate 1b's require_scaler_match() call is skipped, but a loud,
         unmissable warning is printed instead of a quiet pass-through --
         this is not the same as removing the check, it's converting a
         hard-exit into a disclosed, logged override for a specific,
         understood experiment.
       - --force-scaler-mismatch with no --scaler-source is a no-op (there's
         nothing to force past) and prints a note saying so, rather than
         silently doing nothing.
       - --force-scaler-mismatch is REFUSED if --scaler-source is not also
         given -- forcing past a mismatch only makes sense paired with a
         deliberate override, not as a general safety-off switch.
       - Provenance recorded in the output JSON: 'force_scaler_mismatch'
         field records True/False so any result produced this way is
         traceable as such in event_results_*.json, same discipline as
         'conformal' metadata.

Does NOT touch build_dataset.py / train_baseline.py / the scaler-swap math
itself (lines computing X_raw / X_eval under override_scaler are UNCHANGED)
-- this only changes whether Gate 1b's post-hoc consistency check is allowed
to be overridden, and only when explicitly asked to.

Run from ~/dissertation/ with akida_env activated:
    python3 apply_force_scaler_mismatch_patch.py

Hard-refuses (exits nonzero, writes nothing) if any anchor text isn't found
exactly once -- re-run after checking whether the file changed since this
patch was written against the 4 July 2026 snapshot (post-conformal-patch).
"""
import sys

PATH = 'src/evaluation/eval_event_level.py'


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


# ── 1. New flag, plus a refusal if it's used without --scaler-source ────────
old_1 = '''parser.add_argument('--conformal-alpha', type=float, default=0.05,
                    help="Target miscoverage for --conformal: guarantees "
                         "P(future non-seizure window fires) <= alpha "
                         "(default 0.05). First screen also tries 0.01 "
                         "for a tighter guarantee.")
args = parser.parse_args()
assert os.path.exists(args.fbz), f"Model not found: {args.fbz}"
if args.conformal and args.threshold_sweep:
    parser.error('--conformal and --threshold-sweep are mutually exclusive '
                 '-- sweep is exploratory, conformal is the principled '
                 'selection procedure; run them separately.')'''

new_1 = '''parser.add_argument('--conformal-alpha', type=float, default=0.05,
                    help="Target miscoverage for --conformal: guarantees "
                         "P(future non-seizure window fires) <= alpha "
                         "(default 0.05). First screen also tries 0.01 "
                         "for a tighter guarantee.")
parser.add_argument('--force-scaler-mismatch', action='store_true',
                    help="C2 compounding item 2 (amplitude-calibration + "
                         "C2 stacking): Gate 1b hard-refuses any "
                         "--scaler-source override whose scale/shift "
                         "doesn't match the checkpoint's own recorded "
                         "training scaler -- correct behaviour for "
                         "catching an ACCIDENTAL mismatch (the chb06 bug "
                         "class), but it also blocks a DELIBERATE, "
                         "disclosed scale shift for a controlled "
                         "experiment (e.g. stacking sec3b's amplitude-"
                         "calibration scaler on top of an already fine-"
                         "tuned checkpoint). This flag converts that "
                         "hard-exit into a loud, logged override -- it "
                         "does not silence or remove the check, it "
                         "disclosed-bypasses it for this run only. "
                         "Requires --scaler-source; refused otherwise.")
args = parser.parse_args()
assert os.path.exists(args.fbz), f"Model not found: {args.fbz}"
if args.conformal and args.threshold_sweep:
    parser.error('--conformal and --threshold-sweep are mutually exclusive '
                 '-- sweep is exploratory, conformal is the principled '
                 'selection procedure; run them separately.')
if args.force_scaler_mismatch and not args.scaler_source:
    parser.error('--force-scaler-mismatch requires --scaler-source -- '
                 'there is no mismatch to force past without an override '
                 'scaler in the first place.')'''

# ── 2. Gate 1b: allow a disclosed override instead of a hard exit ───────────
old_2 = '''        if expected_scaler is not None:
            if args.longctx or args.g_features:
                _tag = 'Longctx' if args.longctx else 'Candidate G'
                print(f"\\n[Gate 1b] {_tag} run — per-channel scaler; "
                      "require_scaler_match skipped (Gate 2c verified at training).")
            else:
                actual_scaler = override_scaler if args.scaler_source else own_scaler
                require_scaler_match(
                    expected_scaler, actual_scaler,
                    context=f"{args.fbz} (trained) vs. eval input for "
                             f"{args.eval_patient} (this run)",
                )
                print(f"\\n[Gate 1b] Scaler consistency verified against "
                      f"{args.fbz}'s manifest ✓")'''

new_2 = '''        if expected_scaler is not None:
            if args.longctx or args.g_features:
                _tag = 'Longctx' if args.longctx else 'Candidate G'
                print(f"\\n[Gate 1b] {_tag} run — per-channel scaler; "
                      "require_scaler_match skipped (Gate 2c verified at training).")
            elif args.force_scaler_mismatch:
                actual_scaler = override_scaler if args.scaler_source else own_scaler
                print(f"\\n[Gate 1b] *** DELIBERATE OVERRIDE — check bypassed "
                      f"by --force-scaler-mismatch ***")
                print(f"  Model's own recorded training scaler : "
                      f"scale={expected_scaler['scale']:.2f}, "
                      f"shift={expected_scaler['shift']:.2f}")
                print(f"  Scaler actually applied this run      : "
                      f"scale={actual_scaler['scale']:.2f}, "
                      f"shift={actual_scaler['shift']:.2f}")
                print(f"  This is NOT the chb06 dual-scaler bug being "
                      "re-triggered -- it's a disclosed, controlled scale "
                      "shift for a stacking experiment. Treat any result "
                      "from this run as exploratory: the model has never "
                      "seen this input distribution during training.")
            else:
                actual_scaler = override_scaler if args.scaler_source else own_scaler
                require_scaler_match(
                    expected_scaler, actual_scaler,
                    context=f"{args.fbz} (trained) vs. eval input for "
                             f"{args.eval_patient} (this run)",
                )
                print(f"\\n[Gate 1b] Scaler consistency verified against "
                      f"{args.fbz}'s manifest ✓")'''

# ── 3. Record the override in the output JSON for provenance ────────────────
old_3 = """    json.dump({
        'fbz_path': args.fbz,
        'eval_patient': args.eval_patient,
        'spike_threshold': args.spike_threshold,
        'conformal': conformal_info,
        'scaler_source': args.scaler_source if args.scaler_source else own_scaler_path,"""

new_3 = """    json.dump({
        'fbz_path': args.fbz,
        'eval_patient': args.eval_patient,
        'spike_threshold': args.spike_threshold,
        'conformal': conformal_info,
        'force_scaler_mismatch': bool(args.force_scaler_mismatch),
        'scaler_source': args.scaler_source if args.scaler_source else own_scaler_path,"""


if __name__ == '__main__':
    patch_file(PATH, [
        (old_1, new_1),
        (old_2, new_2),
        (old_3, new_3),
    ])
    print("\nAll patches applied successfully.")
    print("\nSanity check:")
    print("  python3 -m py_compile src/evaluation/eval_event_level.py")
    print("\nItem 2 stacking experiment (chb15ft_s256, confirmed distinct "
          "checkpoint, matches the seed used in Methodology_Ledger_"
          "Consolidated.md sec3f):")
    print("  python3 src/evaluation/eval_event_level.py \\\\")
    print("      --fbz results/seizure_model_chb15ft_s256_v2_w4a4.fbz \\\\")
    print("      --eval-patient chb15 \\\\")
    print("      --scaler-source data/processed/chb15_calib120s_scaler.json \\\\")
    print("      --force-scaler-mismatch")
    print("\nRead the result against three numbers: C1 alone (0.000), "
          "C1+amplitude-calib (0.571), C2 alone (0.286, from chb15ft_s123's "
          "threshold-sweep default). Check Gate 0c collapse alongside "
          "sensitivity, not instead of it.")
