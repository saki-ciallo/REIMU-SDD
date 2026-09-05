#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CALLER_DIR="$(pwd -P)"
COMMON_CONFIG="$SCRIPT_DIR/configs/common.yaml"
DRY_RUN=false

if (( $# > 0 )) && [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

run_training() {
  local experiment_config="$1"

  local candidate
  if [[ "$experiment_config" == /* ]]; then
    candidate="$experiment_config"
  else
    for candidate in "$SCRIPT_DIR/$experiment_config" "$CALLER_DIR/$experiment_config"; do
      if [[ -f "$candidate" ]]; then
        break
      fi
    done
  fi
  if [[ -f "$candidate" ]]; then
    experiment_config="$candidate"
  fi
  if [[ ! -f "$experiment_config" ]]; then
    echo "Missing experiment config: $experiment_config" >&2
    return 1
  fi

  echo
  echo "Training optimized implementation"
  echo "  common:     $COMMON_CONFIG"
  echo "  experiment: $experiment_config"
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '  command: '
    printf '%q ' python "$SCRIPT_DIR/train.py" \
      --config "$COMMON_CONFIG" \
      --config "$experiment_config"
    printf '\n'
  else
    python "$SCRIPT_DIR/train.py" \
      --config "$COMMON_CONFIG" \
      --config "$experiment_config"
  fi
}

if (( $# == 0 )); then
  echo "Usage: $0 [--dry-run] EXPERIMENT_CONFIG [EXPERIMENT_CONFIG ...]" >&2
  echo "Example: $0 --dry-run configs/smoke.yaml" >&2
  exit 2
fi

for experiment_config in "$@"; do
  run_training "$experiment_config"
done

# An editable ablation queue can also use explicit calls:
# run_training "configs/experiments/baseline/baseline_attention_l6_freeze_wav2vec2_mhgap.yaml"
# run_training "configs/experiments/baseline/baseline_gdn2_l6_freeze_wav2vec2_mhgap.yaml"
