#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: ./code_optimize/test.sh DATASET [CLI_OPTIONS ...]
       ./code_optimize/test.sh --dataset DATASET [CLI_OPTIONS ...]

Run the test stage through code_optimize/evaluation/<DATASET>.py.

Examples:
  ./code_optimize/test.sh asv19 \
    --model_path code_optimize/outputs/example_run --checkpoint_mode best
  ./code_optimize/test.sh asv21 \
    --model_path code_optimize/outputs/example_run --datasets la --checkpoint_mode last
  ./code_optimize/test.sh asv21 \
    --model_path outputs/example_run --datasets df

All options after DATASET are forwarded to the selected dataset dispatcher.
Adding a new dataset only requires adding its dispatcher under
code_optimize/evaluation/.
EOF
}

if (( $# == 0 )); then
  usage >&2
  exit 2
fi

if [[ "$1" == "--dataset" ]]; then
  if (( $# < 2 )) || [[ -z "$2" ]]; then
    echo "Missing value for --dataset." >&2
    exit 2
  fi
  DATASET="$2"
  shift 2
else
  DATASET="$1"
  shift
fi

if [[ ! "$DATASET" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "Invalid dataset dispatcher name: $DATASET" >&2
  exit 2
fi

CLI_PATH="$SCRIPT_DIR/evaluation/${DATASET}.py"
if [[ ! -f "$CLI_PATH" ]]; then
  echo "Missing dataset CLI: $CLI_PATH" >&2
  echo "Add evaluation/${DATASET}.py or choose an existing dispatcher." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec python "$CLI_PATH" test "$@"
