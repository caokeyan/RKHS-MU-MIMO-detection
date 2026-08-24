#!/bin/bash
set -u
cd "$(dirname "$0")/.."
LOG="extended_results/batch_cdl.log"
PIDF="extended_results/cdl.pid"
{
  echo ""
  echo "===== START $(date '+%Y-%m-%d %H:%M:%S') ====="
} >> "$LOG"

run_one () {
  local ch="$1"
  echo "LAUNCH channel=$ch $(date '+%H:%M:%S')" | tee -a "$LOG"
  python3 -u run_extended_exp.py \
    --mod-order 16 \
    --only realistic \
    --channel "$ch" \
    --n-chan 2 \
    --n-train 2200 \
    --snr-list 0,4,8,12 \
    --save-dir extended_results/qam16 \
    >> "$LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$PIDF"
  wait "$pid"
  local rc=$?
  echo "EXIT channel=$ch rc=$rc $(date '+%H:%M:%S')" | tee -a "$LOG"
  return $rc
}

# cdl_c 由独立进程并行跑；本脚本只负责 cdl_a
run_one cdl_a || exit $?
echo "CDL_A_DONE $(date '+%H:%M:%S')" | tee -a "$LOG"
# 若并行 cdl_c 已结束则汇总，否则只标记 A 完成
if ! pgrep -f 'run_extended_exp.py --mod-order 16 --only realistic --channel cdl_c' >/dev/null; then
  echo "ALL_CDL_DONE $(date '+%H:%M:%S')" | tee -a "$LOG"
else
  echo "WAITING_CDL_C $(date '+%H:%M:%S')" | tee -a "$LOG"
fi
