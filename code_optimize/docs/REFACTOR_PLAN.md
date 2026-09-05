# Refactor plan and completion gates

## Non-negotiable boundary

The legacy model code outside `code_optimize/` is a behavioral reference. The
modular implementation does not import `src`, `addition_loss`, or the parent
`utilis` package. The canonical CPU RawBoost implementation is bundled under
`code_optimize/utilis/` and runs before model input.

## Preserved contracts

1. Input waveform shape is `[batch, samples]`.
2. Frontends return `[batch, sequence, hidden]`.
3. Backbones return hidden states, optional cache/history/attention slots, and
   optional MoE auxiliary loss.
4. Pooling returns `[batch, pooled_size]`.
5. Classifiers return logits and do not own training loss.
6. `ADDModel` supports Hugging Face configuration and checkpoint save/load.
7. Trainer/Accelerate owns BF16 autocast.
8. Cache is disabled by default during training.

## Completed phases

### 1. Configuration and package boundary

- strict duplicate-key YAML loader;
- immutable typed block specifications;
- explicit pipeline, architecture, mixer, feed-forward, pooling, and classifier
  options;
- recursive common-to-experiment merge with whole-component replacement;
- standard Python package metadata.

Gate evidence: configuration tests, wheel build, and legacy model-import search.

### 2. Leaf components

- linear, SincNet, and SSL frontends;
- MHGAP, GAMP, GAPv1, and GAPv2 pooling;
- linear and AMSoftmax classifier heads;
- weighted cross-entropy and focal losses;
- shared dtype-neutral output contracts.

Gate evidence: shape, finite-value, and backward tests for each component.

### 3. Backbone layers

- direct upstream FLA mixer construction;
- upstream GatedMLP and isolated LatentMoE implementation;
- FLA-style pre-norm sequence/feed-forward block;
- cache and MoE auxiliary-loss propagation.

Gate evidence: BF16 forward/backward tests for Attention, Raven, GDN2, and
Mamba3; cache continuation test.

### 4. Architecture orchestration

- baseline;
- looped shared-module recurrence;
- homogeneous HRM;
- heterogeneous HRM.

Gate evidence: explicit gradient schedules and CUDA input/parameter-gradient
tests for all four schedules.

### 5. Top-level model and training

- explicit `backbone_pooling` and `ssl_aasist` pipelines;
- structured experiment settings;
- fixed-waveform dataset loading and collator;
- CUDA RawBoost algorithm 4;
- custom Trainer and CUDA-only runner;
- model report and inference benchmark;
- independent ASV19/ASV21 test entrypoints and scoring entrypoints;
- train/test/score pipeline orchestration.

Gate evidence: complete optimizer step, real eager and compiled Trainer runs,
checkpoint round-trip, model reload prediction smoke, official ASV19/ASV21
scoring smoke, AutoClass loading, and benchmark smoke.

### 6. Release audit

- Ruff lint and format checks;
- complete unittest suite;
- installable wheel;
- no optimized imports from the reference packages;
- reproducible smoke commands and failure records.

The authoritative commands and observed results are maintained in the README
and [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
