#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_PYTHON="${PIPER_SIM_PYTHON:-/mnt/cpfs/users/mrq/emboddied/mujoco/bin/python}"
TRAIN_PYTHON="${PIPER_TRAIN_PYTHON:-/mnt/cpfs/users/mrq/emboddied/openpi/.venv/bin/python}"
case "${1:-}" in
  train|norms|serve|convert|convert-captures) TASK_PYTHON="$TRAIN_PYTHON" ;;
  *) TASK_PYTHON="$SIM_PYTHON" ;;
esac
if [[ ! -x "$TASK_PYTHON" ]]; then
  echo "Python executable not found: $TASK_PYTHON" >&2
  exit 1
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "$TASK_PYTHON" -m piper_titration --config "${PIPER_CONFIG:-$PROJECT_ROOT/configs/chemical_server.json}" "$@"
