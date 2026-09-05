# Optimized ADD Stack

`code_optimize/` is a self-contained implementation of the audio deepfake
detection pipeline. It keeps model construction, dataset handling, training,
dataset-specific evaluation, and reusable utilities inside this directory;
the legacy root `src/` and `utilis/` packages are not required at runtime.

## Pipeline

The standard path is:

```text
waveform -> frontend -> backbone -> pooling -> classifier
```

The available model configuration also includes an optional SSL-to-AASIST
path:

```text
waveform -> SSL frontend -> AASIST -> classifier
```

Backbone schedules cover the ordinary baseline stack as well as looped, HRM,
and heterogeneous HRM variants. Directory-level responsibilities are
documented in [add_system/README.md](add_system/README.md) and its component
README files; architecture and state-flow details are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Extending the Model

The optimized implementation separates configuration, computation, assembly,
and training so a new component can be added at the narrowest boundary that
owns it. Most additions should not modify `ADDModel` or the training runner.

| Addition | Configuration boundary | Model boundary |
| --- | --- | --- |
| frontend | `add_system/configuration/components/frontends.py` | `add_system/models/frontends/` factory |
| pooling method | `add_system/configuration/components/pooling.py` | `add_system/models/pooling/` registry |
| backbone mixer | `add_system/configuration/blocks.py` and mixer config | `add_system/models/backbones/mixers/` factory |
| feed-forward type | `add_system/configuration/blocks.py` and backbone config | `add_system/models/backbones/feedforward/` factory |
| backbone schedule | `ArchitectureType` and backbone config | `add_system/models/backbones/architectures/` factory |
| complete pipeline | `PipelineType`, `ADDConfig`, YAML factory | `add_system/models/add.py` |

The detailed implementation checklist and output contracts are in
[add_system/models/README.md](add_system/models/README.md#extending-the-model).
The corresponding YAML and typed-config wiring is documented in
[add_system/configuration/README.md](add_system/configuration/README.md#adding-configuration-for-a-module).

## Directory Responsibilities

```text
add_system/   model, configuration, data, diagnostics, and training package
configs/      common defaults and experiment YAML overlays
evaluation/   ASVspoof dataset test and score implementations
utilis/       bundled model-independent helpers, including CPU RawBoost
tests/        automated unit, schema, CUDA, and smoke tests
docs/         architecture and maintenance documentation
```

The top-level shell scripts are the normal user-facing entry points. With the
default `run.output_root: outputs`, the optimized runner anchors runtime output
at `code_optimize/outputs/`, including checkpoints, resolved configuration,
test predictions, and scored results. The repository-level `outputs/` belongs
to the legacy implementation and is not used by this package.

## Environment

The intended environment is Python 3.12 with CUDA, BF16 support, PyTorch,
Transformers, Datasets, and FLA installed. The optimized implementation
expects a CUDA device for its model path and does not silently replace FLA
modules with a CPU fallback. No package installation or `build/` directory is
needed for the direct workflow.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py312
```

## Train

Use one common configuration and one experiment overlay. The shell launcher
accepts multiple experiment files and runs them in order:

```bash
./code_optimize/train.sh \
  code_optimize/configs/experiments/baseline/baseline_attention_l6_freeze_wav2vec2_mhgap.yaml
```

For a minimal validation run:

```bash
./code_optimize/train.sh code_optimize/configs/smoke.yaml
```

The resolved configuration and model overview are stored with the run. The
Trainer keeps the best checkpoint according to the configured evaluation
metric and can also retain the final checkpoint.

## Test and Score

Testing and scoring are separate stages. The dataset name is passed to the
generic launcher, while dataset-specific options are forwarded to the
corresponding module under `evaluation/`:

```bash
./code_optimize/test.sh asv19 \
  --model_path code_optimize/outputs/example_run --checkpoint_mode best

./code_optimize/test.sh asv21 \
  --model_path code_optimize/outputs/example_run --datasets la --checkpoint_mode last

./code_optimize/score.sh asv19 \
  --results-dir code_optimize/outputs/example_run

./code_optimize/score.sh asv21 \
  --results-dir code_optimize/outputs/example_run --datasets df
```

Prediction CSV files and metric JSON files are placed in
`<run>/tested_results`; score artifacts are placed below
`<run>/tested_results/scored_results`.

## Full Pipeline

`run_pipeline.sh` runs train, test, and score for each supplied experiment.
Edit its small set of shell variables when running an ablation queue:
`ASV21_DATASET`, `CHECKPOINT_MODE`, `SUBSET`, and
`PIPELINE_STAGES_TEXT`. Use `--dry-run` to inspect commands without executing
them:

```bash
./code_optimize/run_pipeline.sh --dry-run \
  code_optimize/configs/experiments/baseline/baseline_attention_l6_freeze_wav2vec2_mhgap.yaml
```

The dedicated smoke pipeline exercises a short train/test/score path. It is a
plumbing check, not a model-quality evaluation:

```bash
./code_optimize/smoke_run_pipeline.sh
```

## Configuration Migration

The optimized YAML catalog uses structured sections. When the legacy root
configuration catalog changes, regenerate its equivalent overlays with:

```bash
PYTHONPATH=code_optimize python code_optimize/migrate_configs_to_new.py
```

The command preserves optimized-only files by default. Add `--prune` when the
destination must exactly mirror the legacy experiment tree.

## Verification

Run the package tests from the repository root:

```bash
PYTHONPATH=code_optimize python -m unittest discover -s code_optimize/tests -v
```

For formatting and lint checks, use Ruff without creating a cache:

```bash
ruff check --no-cache code_optimize
ruff format --check --no-cache code_optimize
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for known runtime and
upstream-library issues, and [docs/COMPLETION_AUDIT.md](docs/COMPLETION_AUDIT.md)
for the implementation audit.
