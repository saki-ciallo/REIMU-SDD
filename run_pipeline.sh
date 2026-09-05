#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/"

# Shared settings for every ablation experiment below.
COMMON_CONFIG="configs/train_common.yaml"
ASV21_DATASET="both" # la, df, or both
CHECKPOINT_MODE="best" # best, last, or both 实际表现最后的会好一点
SUBSET="eval"
DRY_RUN=false

if [[ ! -f "$COMMON_CONFIG" ]]; then
  echo "Missing common config: $COMMON_CONFIG" >&2
  exit 1
fi

validate_asv21_dataset() {
  local dataset="$1"
  if [[ "$dataset" != "la" && "$dataset" != "df" && "$dataset" != "both" ]]; then
    echo "ASV21 dataset must be la, df, or both; got: $dataset" >&2
    exit 1
  fi
}

validate_checkpoint_mode() {
  local checkpoint_mode="$1"
  if [[ "$checkpoint_mode" != "best" && "$checkpoint_mode" != "last" && "$checkpoint_mode" != "both" ]]; then
    echo "checkpoint mode must be best, last, or both; got: $checkpoint_mode" >&2
    exit 1
  fi
}

validate_stage() {
  local stage="$1"
  if [[ "$stage" != "train" && "$stage" != "test" && "$stage" != "score" ]]; then
    echo "stage must be train, test, or score; got: $stage" >&2
    exit 1
  fi
}

stage_enabled() {
  local target="$1"
  shift
  local stage
  for stage in "$@"; do
    if [[ "$stage" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

resolve_run_dir() {
  local experiment_config="$1"
  python - "$COMMON_CONFIG" "$experiment_config" <<'PY'
import sys
from pathlib import Path

from utilis.config_utils import merge_yaml_configs

config = merge_yaml_configs(*sys.argv[1:])
dataset_key = str(config.get("dataset_key", "asv19"))
task_name = str(config.get("task_name", "need_specify_task_name"))
dataset_prefix = f"{dataset_key}_"
run_name = task_name if task_name.startswith(dataset_prefix) else f"{dataset_prefix}{task_name}"

output_dir = config.get("output_dir")
if output_dir in {None, "trainer_output"}:
    output_dir = Path(str(config.get("output_root", "outputs"))) / run_name

print(Path(str(output_dir)).expanduser().resolve())
PY
}

resolve_gpu_id() {
  local experiment_config="$1"
  python - "$COMMON_CONFIG" "$experiment_config" <<'PY'
import sys

from utilis.config_utils import merge_yaml_configs

config = merge_yaml_configs(*sys.argv[1:])
gpu_id = config.get("gpu_id")
if gpu_id is not None:
    print(gpu_id)
PY
}

resolve_checkpoint_labels() {
  local run_dir="$1"
  local checkpoint_mode="$2"
  python - "$run_dir" "$checkpoint_mode" <<'PY'
import sys

from utilis.model_test_utils import discover_model_variants

for variant in discover_model_variants(sys.argv[1], sys.argv[2]):
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
  if [[ "$DRY_RUN" != "true" ]]; then
    "$@"
  fi
}

run_experiment() {
  local experiment_config="$1"
  shift
  local asv21_dataset="$ASV21_DATASET"
  local checkpoint_mode="$CHECKPOINT_MODE"
  local -a stages=("$@")

  # With no stage arguments, run the complete train -> test -> score pipeline.
  if (( ${#stages[@]} == 0 )); then
    stages=(train test score)
  fi

  if [[ ! -f "$experiment_config" ]]; then
    echo "Missing experiment config: $experiment_config" >&2
    exit 1
  fi
  local stage
  for stage in "${stages[@]}"; do
    validate_stage "$stage"
  done
  validate_asv21_dataset "$asv21_dataset"
  validate_checkpoint_mode "$checkpoint_mode"

  local run_dir
  run_dir="$(resolve_run_dir "$experiment_config")"
  local gpu_id
  gpu_id="$(resolve_gpu_id "$experiment_config")"
  local -a gpu_env=()
  if [[ -n "$gpu_id" ]]; then
    gpu_env=(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_id")
  fi

  echo
  echo "================================================================================"
  echo "Experiment config: $experiment_config"
  echo "Run directory:     $run_dir"
  echo "Stages:            ${stages[*]}"
  echo "ASV21 dataset:     $asv21_dataset"
  echo "Checkpoint mode:   $checkpoint_mode"
  echo "GPU ID:            ${gpu_id:-external/default}"
  echo "================================================================================"

  if stage_enabled train "${stages[@]}"; then
    run_step "train" \
      "${gpu_env[@]}" \
      python train.py \
      --config "$COMMON_CONFIG" \
      --experiment-config "$experiment_config"
  fi

  if stage_enabled test "${stages[@]}"; then
    run_step "test ASV19 LA" \
      "${gpu_env[@]}" \
      python test_asv19.py \
      --model_path "$run_dir" \
      --checkpoint_mode "$checkpoint_mode"

    run_step "test ASV21" \
      "${gpu_env[@]}" \
      python test_asv21.py \
      --model_path "$run_dir" \
      --datasets "$asv21_dataset" \
      --checkpoint_mode "$checkpoint_mode"
  fi

  if stage_enabled score "${stages[@]}"; then
    local -a checkpoint_labels=()
    if [[ "$DRY_RUN" == "true" && ! -d "$run_dir" ]]; then
      checkpoint_labels=(selected_step)
    else
      mapfile -t checkpoint_labels < <(
        resolve_checkpoint_labels "$run_dir" "$checkpoint_mode"
      )
    fi
    if (( ${#checkpoint_labels[@]} == 0 )); then
      echo "No checkpoints selected under $run_dir" >&2
      exit 1
    fi

    local -a asv19_score_command=(
      python score_asv19.py
      --results-dir "$run_dir"
    )
    local checkpoint_label
    for checkpoint_label in "${checkpoint_labels[@]}"; do
      asv19_score_command+=(
        --file-name "asv19la_${checkpoint_label}.csv"
      )
    done
    run_step "score ASV19 LA" "${asv19_score_command[@]}"

    local -a asv21_score_command=(
      python score_asv21.py
      --results-dir "$run_dir"
      --datasets "$asv21_dataset"
      --subset "$SUBSET"
    )
    local dataset_key
    for checkpoint_label in "${checkpoint_labels[@]}"; do
      if [[ "$asv21_dataset" == "both" ]]; then
        for dataset_key in la df; do
          asv21_score_command+=(
            --file-name "asv21${dataset_key}_${checkpoint_label}.csv"
          )
        done
      else
        asv21_score_command+=(
          --file-name "asv21${asv21_dataset}_${checkpoint_label}.csv"
        )
      fi
    done
    run_step "score ASV21" \
      "${asv21_score_command[@]}"
  fi

  echo
  echo "Experiment complete: $run_dir"
}

# Stage selection examples:
# run_experiment "config.yaml"                 # train + test + score
# run_experiment "config.yaml" train           # train only
# run_experiment "config.yaml" test score      # test + score only
# ASV21_DATASET="df" CHECKPOINT_MODE="last" run_experiment "config.yaml" test score

# Ablation queue: by default each experiment completes train -> test -> score.


# algo 0
run_experiment "configs/experiments/finetune/baseline_gdn2_l6_finetune_algo0_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/looped_gdn2_l6_n2_finetune_algo0_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/looped_gdn2_l6_n3_finetune_algo0_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hrm_gdn2_l3_h2l1_finetune_algo0_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hrm_gdn2_l3_h2l2_finetune_algo0_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hrm_gdn2_l3_h2l3_finetune_algo0_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hetero_HALG_l3_h2l1_finetune_algo0_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hetero_HALG_l3_h2l2_finetune_algo0_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hetero_HALG_l3_h2l3_finetune_algo0_wav2vec2_mhgap.yaml"

# algo 4
run_experiment "configs/experiments/finetune/baseline_gdn2_l6_finetune_algo4_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/looped_gdn2_l6_n2_finetune_algo4_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/looped_gdn2_l6_n3_finetune_algo4_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hrm_gdn2_l3_h2l1_finetune_algo4_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hrm_gdn2_l3_h2l2_finetune_algo4_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hrm_gdn2_l3_h2l3_finetune_algo4_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hetero_HALG_l3_h2l1_finetune_algo4_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hetero_HALG_l3_h2l2_finetune_algo4_wav2vec2_mhgap.yaml"
run_experiment "configs/experiments/finetune/hetero_HALG_l3_h2l3_finetune_algo4_wav2vec2_mhgap.yaml"


echo
echo "All configured experiments completed."
