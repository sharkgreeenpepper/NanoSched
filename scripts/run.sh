#!/usr/bin/env bash
set -euo pipefail
NANOSCHED_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
NANOSCHED_PYTHON=${NANOSCHED_PYTHON:-/data1/fangdongqing/infra/runtime/nonprefix-kv-20261008/bin/python}
NANOSCHED_FLASH_ROOT=${NANOSCHED_FLASH_ROOT:-/data1/fangdongqing/git/memo-kv-serve/.aris/epicache_c_20261013/flash_build/flash_attn-2.7.4.post1}
NANOSCHED_CUDA_COMPAT=${NANOSCHED_CUDA_COMPAT:-/data1/fangdongqing/infra/runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat}
export LD_LIBRARY_PATH="$NANOSCHED_CUDA_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$NANOSCHED_ROOT/.runtime-deps:$NANOSCHED_FLASH_ROOT:$NANOSCHED_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec "$NANOSCHED_PYTHON" "$@"
