#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: ./code_optimize/score.sh DATASET [CLI_OPTIONS ...]
       ./code_optimize/score.sh --dataset DATASET [CLI_OPTIONS ...]

Run the score stage through code_optimize/evaluation/<DATASET>.py.

Examples:
  ./code_optimize/score.sh asv19 \
    --results-dir code_optimize/outputs/example_run --file-name asv19la_100.csv
  ./code_optimize/score.sh asv21 \
    --results-dir code_optimize/outputs/example_run --datasets la
  ./code_optimize/score.sh asv21 \
    --results-dir code_optimize/outputs/example_run --datasets df

All options after DATASET are forwarded to the selected dataset dispatcher.
Dataset tracks such as LA or DF are intentionally owned by that dispatcher,
so this wrapper does not need to be changed when a new dataset or track is
added.
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
exec python "$CLI_PATH" score "$@"
