#!/bin/bash
# Runs baselines_residuo.py with automatic resume if the process is killed for lack of memory
# Usage: scripts/rodar_revisao17.sh [drifter|year]
cd "$(dirname "$0")/.." || exit 1
CVV=${1:-drifter}; LOG=resultados/baselines_residuo_${CVV}.log
for i in $(seq 1 15); do
  CV=$CVV SKIP_STATS=1 python3 scripts/baselines_residuo.py >> $LOG 2>&1 && grep -q FIM $LOG && { echo "ok $CVV"; exit 0; }
  echo "attempt $i failed ($?), resuming" >> $LOG; sleep 20
done
echo "failed $CVV"
