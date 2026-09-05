#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/"

RUN_NAME="asv19_baseline_atten_l12"
DATASETS="df" # la, df, both
SUBSET="eval"

python score_asv21.py \
  --results-dir "asv19_baseline_attention_l6_freeze_wavlm_mhgap" \
  --datasets "$DATASETS" \
  --subset "$SUBSET"


python score_asv21.py \
  --results-dir "asv19_baseline_attention_l6_freeze_wavlm_plus_mhgap" \
  --datasets "$DATASETS" \
  --subset "$SUBSET"
