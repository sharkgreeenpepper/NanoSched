#!/usr/bin/env bash
set -euo pipefail
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
if [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; then exit 1; fi
nvidia-smi -i 0 --query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw --format=csv -lms 500 > reports/phase01_20261022/followup_gpu_telemetry.csv &
NANOSCHED_MONITOR_PID=$!
trap 'kill "$NANOSCHED_MONITOR_PID" 2>/dev/null || true; wait "$NANOSCHED_MONITOR_PID" 2>/dev/null || true' EXIT
export CUDA_VISIBLE_DEVICES=0
bash scripts/run.sh -m benchmarks.replay --engine-root /data1/fangdongqing/git/NanoSched-phase1-8ef7814 --policy decode_aware --eager --output reports/phase01_20261022/final_competition_decode_aware_eager --repeats 3 > reports/phase01_20261022/final_competition_decode_aware_eager.stdout 2>&1
for NANOSCHED_CASE in upstream adapted optimized; do
    case "$NANOSCHED_CASE" in
        upstream) NANOSCHED_ENGINE=/data1/fangdongqing/git/NanoSched-baseline-bb823b3; NANOSCHED_POLICY=prefill_first ;;
        adapted) NANOSCHED_ENGINE=/data1/fangdongqing/git/NanoSched-baseline-phase0; NANOSCHED_POLICY=prefill_first ;;
        optimized) NANOSCHED_ENGINE=/data1/fangdongqing/git/NanoSched-phase1-8ef7814; NANOSCHED_POLICY=decode_aware ;;
    esac
    bash scripts/run.sh -m benchmarks.replay --engine-root "$NANOSCHED_ENGINE" --policy "$NANOSCHED_POLICY" --eager --temperature 1 --output "reports/phase01_20261022/native_${NANOSCHED_CASE}_eager" --repeats 3 > "reports/phase01_20261022/native_${NANOSCHED_CASE}_eager.stdout" 2>&1
done
if bash scripts/run.sh -m benchmarks.replay --engine-root /data1/fangdongqing/git/NanoSched-phase1-8ef7814 --policy decode_aware --workload-file reports/phase01_20261022/profile_workload.json --profile --warmup-runs 1 --repeats 1 --output reports/phase01_20261022/profiler > reports/phase01_20261022/profiler.stdout 2>&1; then
    printf '0\n' > reports/phase01_20261022/profiler.exit_code
else
    printf '%s\n' "$?" > reports/phase01_20261022/profiler.exit_code
fi
for NANOSCHED_MODE in eager graph; do
    NANOSCHED_FLAGS=()
    if [[ "$NANOSCHED_MODE" == eager ]]; then NANOSCHED_FLAGS+=(--eager); fi
    if bash scripts/run.sh -m benchmarks.correctness --policy decode_aware "${NANOSCHED_FLAGS[@]}" --output "reports/phase01_20261022/final_correctness_${NANOSCHED_MODE}.json" > "reports/phase01_20261022/final_correctness_${NANOSCHED_MODE}.stdout" 2>&1; then
        printf '0\n' > "reports/phase01_20261022/final_correctness_${NANOSCHED_MODE}.exit_code"
    else
        printf '%s\n' "$?" > "reports/phase01_20261022/final_correctness_${NANOSCHED_MODE}.exit_code"
    fi
done
