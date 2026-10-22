#!/usr/bin/env bash
# Run sequentially on a single idle GPU; never starts workers on other devices.
set -euo pipefail
NANOSCHED_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
NANOSCHED_OUTPUT=${1:?Pass a fresh results directory}
NANOSCHED_BASELINE=${2:?Pass the pinned Phase 0 worktree}
NANOSCHED_GPU=${CUDA_VISIBLE_DEVICES:?Set one idle physical GPU index}
if [[ ! "$NANOSCHED_GPU" =~ ^[0-9]+$ ]]; then
    printf '%s\n' 'CUDA_VISIBLE_DEVICES must specify one physical index' >&2
    exit 1
fi
if [[ -n "$(nvidia-smi -i "$NANOSCHED_GPU" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    printf '%s\n' 'Selected GPU has an active compute process; refusing to start' >&2
    exit 1
fi
mkdir "$NANOSCHED_OUTPUT"
nvidia-smi -i "$NANOSCHED_GPU" --query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw --format=csv -lms 500 > "$NANOSCHED_OUTPUT/gpu_telemetry.csv" &
NANOSCHED_MONITOR_PID=$!
trap 'kill "$NANOSCHED_MONITOR_PID" 2>/dev/null || true; wait "$NANOSCHED_MONITOR_PID" 2>/dev/null || true' EXIT
for NANOSCHED_MODE in eager graph; do
    NANOSCHED_FLAGS=()
    if [[ "$NANOSCHED_MODE" == eager ]]; then NANOSCHED_FLAGS+=(--eager); fi
    for NANOSCHED_WORKLOAD in competition short long; do
        for NANOSCHED_POLICY in prefill_first decode_aware; do
            NANOSCHED_ENGINE="$NANOSCHED_ROOT"
            if [[ "$NANOSCHED_POLICY" == prefill_first ]]; then NANOSCHED_ENGINE="$NANOSCHED_BASELINE"; fi
            NANOSCHED_NAME="${NANOSCHED_WORKLOAD}_${NANOSCHED_POLICY}_${NANOSCHED_MODE}"
            printf '%s\n' "Running $NANOSCHED_NAME"
            bash "$NANOSCHED_ROOT/scripts/run.sh" -m benchmarks.replay --engine-root "$NANOSCHED_ENGINE" \
                --policy "$NANOSCHED_POLICY" --workload "$NANOSCHED_WORKLOAD" \
                --output "$NANOSCHED_OUTPUT/$NANOSCHED_NAME" --repeats 3 --warmup-runs 1 \
                --temperature 0 --budget 1024 --decode-prefill-budget 1024 "${NANOSCHED_FLAGS[@]}" \
                > "$NANOSCHED_OUTPUT/$NANOSCHED_NAME.stdout" 2>&1
        done
    done
done
