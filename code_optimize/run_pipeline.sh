#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_OPTIMIZE_ROOT="$SCRIPT_DIR"
CALLER_DIR="$(pwd -P)"
COMMON_CONFIG="$CODE_OPTIMIZE_ROOT/configs/common.yaml"
ASV21_DATASET="la" # la, df, or both
CHECKPOINT_MODE="best" # best, last, or both
SUBSET="eval"
PIPELINE_STAGES_TEXT="train,test,score" # comma-separated
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

if [[ ! -f "$COMMON_CONFIG" ]]; then
  echo "Missing common config: $COMMON_CONFIG" >&2
  exit 1
fi

cd "$CODE_OPTIMIZE_ROOT"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
IFS=',' read -r -a PIPELINE_STAGES <<< "$PIPELINE_STAGES_TEXT"

validate_stage() {
  case "$1" in
    train|test|score) ;;
    *) echo "stage must be train, test, or score; got: $1" >&2; exit 1 ;;
  esac
}

validate_dataset() {
  case "$1" in
    la|df|both) ;;
    *) echo "ASV21 dataset must be la, df, or both; got: $1" >&2; exit 1 ;;
  esac
}

validate_checkpoint_mode() {
  case "$1" in
    best|last|both) ;;
    *) echo "checkpoint mode must be best, last, or both; got: $1" >&2; exit 1 ;;
  esac
}

stage_enabled() {
  local target="$1"
  shift
  local stage
  for stage in "$@"; do
    [[ "$stage" == "$target" ]] && return 0
  done
  return 1
}

resolve_experiment_config() {
  local requested="$1"
  local candidate

  if [[ "$requested" == /* ]]; then
    if [[ -f "$requested" ]]; then
      candidate="$requested"
    else
      echo "Missing experiment config: $requested" >&2
      return 1
    fi
  else
    for candidate in \
      "$CODE_OPTIMIZE_ROOT/$requested" \
      "$CALLER_DIR/$requested"; do
      if [[ -f "$candidate" ]]; then
        break
      fi
    done
    if [[ ! -f "$candidate" ]]; then
      echo "Missing experiment config: $requested" >&2
      return 1
    fi
  fi

  local directory filename
  directory="$(cd "$(dirname "$candidate")" && pwd -P)"
  filename="$(basename "$candidate")"
  printf '%s/%s\n' "$directory" "$filename"
}

resolve_run_dir() {
  python - "$SCRIPT_DIR" "$COMMON_CONFIG" "$1" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from add_system.configuration.experiments import resolve_experiment

settings = resolve_experiment(sys.argv[2:]).build_settings()
print(settings.output_dir.resolve())
PY
}

resolve_checkpoint_labels() {
  python - "$SCRIPT_DIR" "$1" "$2" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from utilis.model_test_utils import discover_model_variants

for variant in discover_model_variants(sys.argv[2], sys.argv[3]):
    print(variant.step_label)
PY
}

run_step() {
  local stage="$1"
  shift
  echo
  echo "[$stage]"
  printf '  '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != true ]]; then
    "$@"
  fi
}

run_experiment() {
  local experiment_config
  experiment_config="$(resolve_experiment_config "$1")" || exit 1
  local -a stages=("${PIPELINE_STAGES[@]}")

  local stage
  for stage in "${stages[@]}"; do
    validate_stage "$stage"
  done
  validate_dataset "$ASV21_DATASET"
  validate_checkpoint_mode "$CHECKPOINT_MODE"

  local run_dir
  run_dir="$(resolve_run_dir "$experiment_config")"
  echo
  echo "================================================================================"
  echo "Experiment config: $experiment_config"
  echo "Run directory:     $run_dir"
  echo "Stages:            ${stages[*]}"
  echo "ASV21 dataset:     $ASV21_DATASET"
  echo "Checkpoint mode:   $CHECKPOINT_MODE"
  echo "================================================================================"

  if stage_enabled train "${stages[@]}"; then
    run_step "train" python "$CODE_OPTIMIZE_ROOT/train.py" \
      --config "$COMMON_CONFIG" \
      --config "$experiment_config"
  fi

  if stage_enabled test "${stages[@]}"; then
    run_step "test ASV19 LA" "$SCRIPT_DIR/test.sh" asv19 \
      --model_path "$run_dir" \
      --checkpoint_mode "$CHECKPOINT_MODE"
    run_step "test ASV21" "$SCRIPT_DIR/test.sh" asv21 \
      --model_path "$run_dir" \
      --datasets "$ASV21_DATASET" \
      --checkpoint_mode "$CHECKPOINT_MODE"
  fi

  if stage_enabled score "${stages[@]}"; then
    local -a checkpoint_labels=()
    if [[ "$DRY_RUN" == true && ! -d "$run_dir" ]]; then
      checkpoint_labels=(selected_step)
    else
      mapfile -t checkpoint_labels < <(
        resolve_checkpoint_labels "$run_dir" "$CHECKPOINT_MODE"
      )
    fi
    if (( ${#checkpoint_labels[@]} == 0 )); then
      echo "No checkpoints selected under $run_dir" >&2
      exit 1
    fi

    local -a asv19_command=("$SCRIPT_DIR/score.sh" asv19 --results-dir "$run_dir")
    local -a asv21_command=(
      "$SCRIPT_DIR/score.sh"
      asv21
      --results-dir "$run_dir"
      --datasets "$ASV21_DATASET"
      --subset "$SUBSET"
    )
    local checkpoint_label dataset_key
    for checkpoint_label in "${checkpoint_labels[@]}"; do
      asv19_command+=(--file-name "asv19la_${checkpoint_label}.csv")
      if [[ "$ASV21_DATASET" == both ]]; then
        for dataset_key in la df; do
          asv21_command+=(--file-name "asv21${dataset_key}_${checkpoint_label}.csv")
        done
      else
        asv21_command+=(
          --file-name "asv21${ASV21_DATASET}_${checkpoint_label}.csv"
        )
      fi
    done
    run_step "score ASV19 LA" "${asv19_command[@]}"
    run_step "score ASV21" "${asv21_command[@]}"
  fi

  echo
  echo "Experiment complete: $run_dir"
}

if (( $# == 0 )); then
  echo "Usage: $0 [--dry-run] EXPERIMENT_CONFIG [EXPERIMENT_CONFIG ...]" >&2
  echo "Example: $0 --dry-run configs/smoke.yaml" >&2
  exit 2
fi

for experiment_config in "$@"; do
  run_experiment "$experiment_config"
done
