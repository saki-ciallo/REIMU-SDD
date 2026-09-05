#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/"

COMMON_CONFIG="configs/train_common.yaml"

run_training() {
  local experiment_config="$1"
  echo "Starting experiment: $experiment_config"
  python train.py \
    --config "$COMMON_CONFIG" \
    --experiment-config "$experiment_config"
}

# Baseline
# run_training "configs/experiments/baseline_attention_l12_sincnet.yaml"
# run_training "configs/experiments/baseline_attention_l12.yaml"

# Pooling ablations (same linear frontend and Attention backbone).
# run_training "configs/experiments/baseline_attention_l12_gamp.yaml"
# run_training "configs/experiments/baseline_attention_l12_gapv1.yaml"

# run_training "configs/experiments/baseline_raven_l12.yaml"
# run_training "configs/experiments/baseline_gdn2_l12.yaml"

# Looped: shared 12-layer module, N=2 total cycles, last K=1 cycle with gradients.
# run_training "configs/experiments/looped_attention_l12_n2_k1.yaml"
# run_training "configs/experiments/looped_raven_l12_n2_k1.yaml"
# run_training "configs/experiments/looped_gdn2_l12_n2_k1.yaml"

# HRM: independent 6-layer H/L modules, H2L3 schedule, last K=2 module calls with gradients.
# run_training "configs/experiments/hrm_attention_l6_h2l3_k2.yaml"
# run_training "configs/experiments/hrm_raven_l6_h2l3_k2.yaml"
# run_training "configs/experiments/hrm_gdn2_l6_h2l3_k2.yaml"

# Heterogeneous HRM: independent H=Attention and L=GDN2 module topologies.
# run_training "configs/experiments/heterogeneous_hrm_h_attention_l_gdn2_l6_h2l3_k2.yaml"

# run_training "configs/experiments/mamba3_l12.yaml"

# SSL frontend ablations (all emit 768-dimensional features).
# run_training "configs/experiments/baseline_attention_l12_wav2vec2_base.yaml"
# run_training "configs/experiments/baseline_attention_l12_hubert_base_ls960.yaml"
# run_training "configs/experiments/baseline_attention_l12_wavlm_base.yaml"
# run_training "configs/experiments/baseline_attention_l12_wavlm_base_plus.yaml"
