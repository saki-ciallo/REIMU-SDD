#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/"

DATASETS="df" # la, df, both
CHECKPOINT_MODE="both" # best, last, both

python test_asv21.py \
  --model_path "asv19_baseline_attention_l6_freeze_hubert_mhgap" \
  --datasets "df" \
  --checkpoint_mode "$CHECKPOINT_MODE"

python test_asv21.py \
  --model_path "asv19_baseline_attention_l6_freeze_wavlm_mhgap" \
  --datasets "df" \
  --checkpoint_mode "$CHECKPOINT_MODE"

python test_asv21.py \
  --model_path "asv19_baseline_attention_l6_freeze_wavlm_plus_mhgap" \
  --datasets "df" \
  --checkpoint_mode "$CHECKPOINT_MODE"
