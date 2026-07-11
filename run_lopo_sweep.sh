#!/usr/bin/env bash
# run_lopo_sweep.sh  (v5 — fixes: cleanup now runs on failure too)
#
# Full 15-fold LOPO sweep, end to end. Run from ~/dissertation/ with
# akida_env activated.
#
# v5 change vs. v4: cleanup was the LAST thing inside run_fold(), after a
# successful eval_event_level.py call. When chb06/chb07/chb09 all got
# OOM-killed AT that eval step (separate bug, fixed by
# apply_lopo_eval_batching_patch.py -- apply that too before re-running),
# run_fold() returned 1 before ever reaching cleanup, so each failed fold's
# ~10GB+ of transient artifacts (pooled dataset + full-recording eval set)
# was never reclaimed. Three failures in a row silently drained the disk
# from 26G free down to 0, which is what then broke chb18 and chb20 on
# "No space left on device" -- a direct downstream consequence, not a
# separate problem. Cleanup is now called unconditionally in the outer
# loop, after run_fold() returns, regardless of its exit status.
#
# Usage:
#   bash run_lopo_sweep.sh                  # all 15 patients
#   bash run_lopo_sweep.sh chb06 chb07 chb09 chb18 chb20   # resume remaining

set -uo pipefail   # NOT -e: a bad fold must not kill the sweep

ROSTER=(chb01 chb02 chb03 chb05 chb06 chb07 chb09 chb10 chb11 chb13 chb15 chb16 chb18 chb19 chb20)
PATIENTS=("${@:-${ROSTER[@]}}")

mkdir -p results/lopo_logs

quietpy() {
  python3 "$@" 2> >(grep -Ev 'cuDNN factory|cuBLAS factory|computation placer already registered|InitializeLog' >&2)
}

# ── Preflight: every ROSTER patient needs raw windows, regardless of which
#    subset of folds this invocation runs (any fold's pool can draw on any
#    of the other 14) ──────────────────────────────────────────────────────
echo "=== Preflight: checking raw windows for all ${#ROSTER[@]} roster patients ==="
for p in "${ROSTER[@]}"; do
  if [[ -f "data/processed/${p}_X.npy" && -f "data/processed/${p}_y.npy" ]]; then
    echo "  ${p}: OK"
  else
    echo "  ${p}: MISSING -- regenerating via preprocess.py"
    time quietpy src/preprocessing/preprocess.py --patient "${p}" \
      || { echo "FAIL: could not regenerate raw windows for ${p} -- check data/raw/chbmit/.../${p} still has its EDFs on disk"; exit 1; }
  fi
done
echo "=== Preflight complete ==="
echo

# ── Per-fold logic as a function: `return` (not `continue`) works correctly
#    no matter how this function's output is redirected. Cleanup is NOT
#    called from inside here anymore -- see cleanup_fold() below, called
#    unconditionally from the outer loop instead. ───────────────────────────
run_fold() {
  local p="$1"
  echo "=== $(date -u) fold ${p} ==="
  echo "Disk before fold: $(df -h . | tail -1 | awk '{print $4" avail ("$5" used)"}')"

  echo "--- [1/4] build_dataset_lopo_fold.py ---"
  time quietpy src/preprocessing/build_dataset_lopo_fold.py --leave-out "${p}" --us-ratio 5 \
    || { echo "FAIL at build_dataset_lopo_fold.py for ${p}"; return 1; }

  echo "--- [2/4] build_lopo_eval_set.py (skip if already built) ---"
  if [[ -f "data/processed/${p}_dataset_lopo_full.npz" ]]; then
    echo "  already exists, skipping"
  else
    time quietpy src/preprocessing/build_lopo_eval_set.py --patient "${p}" \
      || { echo "FAIL at build_lopo_eval_set.py for ${p}"; return 1; }
  fi

  echo "--- [3/4] train_baseline.py --pool-tag lopo_${p} ---"
  time quietpy src/models/train_baseline.py --model-version 2 \
    --multi-patient --pool-tag "lopo_${p}" \
    || { echo "FAIL at train_baseline.py for ${p}"; return 1; }

  echo "--- [4/4] convert_to_snn.py + eval_event_level.py --lopo-full ---"
  time quietpy src/models/convert_to_snn.py --model-version 2 \
    --patient "multi_lopo_${p}" --base "multi_lopo_${p}" --eval-patient "${p}" \
    || { echo "FAIL at convert_to_snn.py for ${p}"; return 1; }

  time quietpy src/evaluation/eval_event_level.py \
    --fbz "results/seizure_model_multi_lopo_${p}_v2_w4a4.fbz" \
    --eval-patient "${p}" --lopo-full \
    || { echo "FAIL at eval_event_level.py for ${p}"; return 1; }

  echo "=== fold ${p} DONE ==="
  return 0
}

# ── Unconditional cleanup — called after EVERY fold attempt, success or
#    failure. Only removes fold-specific transients (never reused by any
#    other fold); the results JSON, if it was produced, is untouched. ──────
cleanup_fold() {
  local p="$1"
  echo "--- [cleanup] removing this fold's transient artifacts (regardless of outcome) ---"
  rm -f "data/processed/multi_lopo_${p}_dataset_ann.npz"
  rm -f "data/processed/multi_lopo_${p}_scaler.json"
  rm -f "data/processed/${p}_dataset_lopo_full.npz"
  echo "Disk after cleanup: $(df -h . | tail -1 | awk '{print $4" avail ("$5" used)"}')"
}

echo "=== LOPO sweep: ${#PATIENTS[@]} fold(s): ${PATIENTS[*]} ==="
echo

for p in "${PATIENTS[@]}"; do
  LOG="results/lopo_logs/${p}.log"
  echo ">>> Fold: leave-out=${p}  (log: ${LOG})"
  run_fold "${p}" 2>&1 | tee "${LOG}"
  fold_status="${PIPESTATUS[0]}"
  if [[ "${fold_status}" -ne 0 ]]; then
    echo "  (fold ${p} did not complete cleanly -- see ${LOG})"
  fi
  cleanup_fold "${p}" 2>&1 | tee -a "${LOG}"
  echo
done

echo "=== Sweep complete. Aggregate with: ==="
echo "  python3 src/evaluation/aggregate_lopo_results.py"
