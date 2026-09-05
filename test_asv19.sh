#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/"

MODEL_PATH="asv19_baseline_attention_l12_freeze_wav2vec2"
CHECKPOINT_MODE="both" # best, last, or both

python test_asv19.py \
  --model_path "$MODEL_PATH" \
  --checkpoint_mode "$CHECKPOINT_MODE"
