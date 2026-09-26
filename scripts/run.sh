#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LIST="${PIPER_GPU_LIST:-4}"
ARGS=()
while (($#)); do
  case "$1" in
    --gpu-list)
      [[ $# -ge 2 ]] || { echo '--gpu-list requires a value' >&2; exit 2; }
      GPU_LIST="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Use --gpu-list 4 or 4,5,6,7' >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
declare -A SEEN=()
for id in "${GPU_IDS[@]}"; do
  [[ "$id" =~ ^[4-7]$ && -z "${SEEN[$id]:-}" ]] || { echo 'Allowed unique GPU IDs: 4,5,6,7' >&2; exit 2; }
  SEEN[$id]=1
done
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/data/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/nas/share/home/lzk/mrq/robot/openpi_data}"
export HF_HOME="${HF_HOME:-/mnt/nas/share/home/lzk/mrq/robot/cache/huggingface}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="${PIPER_CONFIG:-$PROJECT_ROOT/configs/chemical_server.json}"
if [[ -n "${PIPER_TRAIN_PYTHON:-}" ]]; then
  exec "$PIPER_TRAIN_PYTHON" -m piper_titration --config "$CONFIG" "${ARGS[@]}"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "${PIPER_CONDA_ENV:-songling}" ]]; then
  exec python -m piper_titration --config "$CONFIG" "${ARGS[@]}"
else
  exec conda run --no-capture-output -n "${PIPER_CONDA_ENV:-songling}" python -m piper_titration --config "$CONFIG" "${ARGS[@]}"
fi
