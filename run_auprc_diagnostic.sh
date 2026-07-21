#!/usr/bin/env bash
# Re-scoring only -- no new training/conversion. Run from ~/dissertation
# with akida_env active. 6 runs total: C1 (frozen pool) + C2 (per-patient
# ft, s256) on each of chb13/chb15/chb16.
#
# C1 checkpoint confirmed via `ls results/*.fbz | grep multi` this session:
# results/seizure_model_multi_v2_w4a4.fbz (plain, no _lopo_/_dann_/ft_
# suffix -- the frozen chb01/02/05-pool checkpoint). The earlier
# "..._noz_..." filename in this script was wrong -- never existed on
# disk, only an inherited docstring example.
set -e

for p in chb13 chb15 chb16; do
    echo "=== C1 (frozen pool) on $p ==="
    python3 src/evaluation/eval_event_level.py \
        --fbz results/seizure_model_multi_v2_w4a4.fbz \
        --eval-patient $p

    echo "=== C2 (${p}ft_s256) on $p ==="
    python3 src/evaluation/eval_event_level.py \
        --fbz results/seizure_model_${p}ft_s256_v2_w4a4.fbz \
        --eval-patient $p
done
