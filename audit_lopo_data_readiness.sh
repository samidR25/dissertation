#!/usr/bin/env bash
# audit_lopo_data_readiness.sh
# ==============================
# One-pass inventory of what each LOPO roster patient actually has on disk,
# so gaps can be fixed in one batch instead of discovered one at a time
# mid-sweep. Checks three things per patient:
#   1. Raw EDF directory (data/raw/chbmit/.../<patient>/) — non-empty?
#   2. Raw windows (_X.npy/_y.npy in data/processed/) — present?
#   3. Per-patient scaler (_scaler.json) — present? (needed by
#      build_lopo_eval_set.py regardless of raw-window status)
#
# Prints a table, then a ready-to-paste block of `aws s3 sync` commands
# (same pattern already used in this project for chb06/chb11) for any
# patient missing raw EDFs.
#
# Usage:
#   bash audit_lopo_data_readiness.sh

ROSTER=(chb01 chb02 chb03 chb05 chb06 chb07 chb09 chb10 chb11 chb13 chb15 chb16 chb18 chb19 chb20)
RAW_BASE="data/raw/chbmit/physionet.org/files/chbmit/1.0.0"

printf "%-8s %-12s %-12s %-12s\n" "Patient" "RawEDF" "RawWindows" "Scaler"
printf "%-8s %-12s %-12s %-12s\n" "-------" "------" "----------" "------"

missing_edf=()

for p in "${ROSTER[@]}"; do
  if [[ -d "${RAW_BASE}/${p}" ]] && [[ -n "$(ls -A "${RAW_BASE}/${p}" 2>/dev/null)" ]]; then
    edf_status="OK"
  else
    edf_status="MISSING"
    missing_edf+=("${p}")
  fi

  if [[ -f "data/processed/${p}_X.npy" && -f "data/processed/${p}_y.npy" ]]; then
    win_status="OK"
  else
    win_status="missing"
  fi

  if [[ -f "data/processed/${p}_scaler.json" ]]; then
    scaler_status="OK"
  else
    scaler_status="missing"
  fi

  printf "%-8s %-12s %-12s %-12s\n" "${p}" "${edf_status}" "${win_status}" "${scaler_status}"
done

echo
if [[ ${#missing_edf[@]} -eq 0 ]]; then
  echo "All ${#ROSTER[@]} patients have raw EDFs on disk. Raw-window/scaler"
  echo "gaps (if any, see 'missing' above) will be regenerated automatically"
  echo "by run_lopo_sweep.sh's preflight step from these EDFs."
else
  echo "${#missing_edf[@]} patient(s) missing raw EDFs entirely: ${missing_edf[*]}"
  echo
  echo "Re-download commands (same pattern already used for chb06/chb11):"
  for p in "${missing_edf[@]}"; do
    echo "  aws s3 sync s3://physionet-open/chbmit/1.0.0/${p}/ ${RAW_BASE}/${p}/ --no-sign-request"
  done
  echo
  echo "Run these (can run sequentially or in parallel with GNU parallel, as"
  echo "done in the original Phase 1c download session), THEN re-run:"
  echo "  bash audit_lopo_data_readiness.sh     # confirm all OK"
  echo "  bash run_lopo_sweep.sh chb10          # resume the smoke test"
fi

echo
echo "--- Disk space check ---"
df -h data/raw data/processed 2>/dev/null | tail -n +1
echo
echo "NOTE: raw EDFs run ~1.4-1.7GB/patient; some patients' raw windowed"
echo "arrays (_X.npy) have been as large as ~5.4GB (chb01). Re-downloading"
echo "+ re-preprocessing all missing patients simultaneously could need"
echo "tens of GB free. If space is tight, download/preprocess in smaller"
echo "batches rather than all at once -- worth checking 'df -h' output"
echo "above against how many patients are actually missing before running"
echo "the full batch of aws s3 sync commands."
