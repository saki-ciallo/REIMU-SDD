#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/"

RUN_NAME="asv19_baseline_attention_l12_linear"

python score_asv19.py \
  --results-dir "asv19_baseline_attention_l12_freeze_wav2vec2"
  # --file-name asv19la_3176.csv
