# Completion audit

This audit maps the requested refactor outcomes to current artifacts and
verification evidence. It is intentionally narrower than a general “no errors
found” statement.

## Requirement evidence

| Requirement | Authoritative implementation | Verification |
|---|---|---|
| Keep model code isolated | `code_optimize/` package, configs, tests, and docs | no imports from legacy `src`, `addition_loss`, or parent `utilis`; CPU RawBoost is bundled locally |
| Separate model component families | `models/frontends`, `backbones`, `pooling`, `classifier`, `aasist` | component shape/backward tests |
| Separate baseline, looped, HRM, heterogeneous HRM | one class per file under `backbones/architectures` | architecture gradient schedule and CUDA backward tests |
| Make mixer/FFN combinations extensible | `backbones/mixers/factory.py`, `feedforward/factory.py`, immutable `BlockSpec` | Attention/Raven/GDN2/Mamba3 and MLP/MoE smoke |
| Improve experiment and config loading | strict YAML loader, typed runtime settings, nested HF configs | duplicate/unknown-key, dimension, merge, and round-trip tests |
| Preserve HF model workflow | `ADDConfig`, `ADDModel`, explicit AutoClass registration | exact checkpoint logits and AutoModel load |
| Keep training concerns outside model | `training/losses.py`, `trainer.py`, `run.py` | weighted CE, focal, AMSoftmax, aux-loss and Trainer smoke |
| Keep the full experiment lifecycle inside `code_optimize` | `train.py`, `test.sh`, `score.sh`, `evaluation/asv19.py`, `evaluation/asv21.py`, and `run_pipeline.sh` | isolated train/test/score smoke and official scoring |
| Make startup behavior observable | `run_manifest.json` and startup phase timers | real smoke manifest with model/data/Trainer timings |
| Provide performance measurement | `diagnostics/benchmark.py`, `benchmark.py` | eager/default/max-autotune benchmark JSON |
| Use current Python/PyTorch facilities carefully | Python 3.12 enums/dataclasses/type aliases; CUDA events; grouped MM; FFT RawBoost | Ruff, compileall, wheel, CUDA tests |
| Record failed attempts and resolutions | `docs/TROUBLESHOOTING.md` | entries include symptom, root cause, final action, and evidence |
| Provide maintainable user documentation | README plus architecture, decisions, troubleshooting, and this audit | local links and documented commands |

## Final verification commands

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py312

ruff check --no-cache code_optimize
ruff format --check --no-cache code_optimize
python -m compileall -q \
  code_optimize/add_system code_optimize/utilis \
  code_optimize/train.py code_optimize/benchmark.py

PYTHONPATH=code_optimize \
  python -m unittest discover -s code_optimize/tests -v

python code_optimize/train.py \
  --config code_optimize/configs/common.yaml \
  --config code_optimize/configs/smoke.yaml

python code_optimize/train.py \
  --config code_optimize/configs/common.yaml \
  --config code_optimize/configs/smoke.yaml \
  --config code_optimize/configs/smoke_compile.yaml

./code_optimize/test.sh asv19 \
  --model_path /tmp/add_system_smoke/asv19_optimize_smoke \
  --checkpoint_mode last --max_predict_samples 2

./code_optimize/score.sh asv19 \
  --results-dir outputs/example_run --file-name asv19la_3176.csv

python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /tmp/add_system_wheels ./code_optimize

rg -n --glob '*.py' \
  '(from|import) (src|addition_loss)' code_optimize
```

## Observed final results

- Ruff: passed, all files formatted.
- Compileall: passed.
- Unit/CUDA suite: 51 tests and 194 subtests passed.
- Eager Trainer: two train steps plus evaluation and checkpoint save passed.
- Default Inductor Trainer: two train steps plus evaluation and checkpoint save
  passed.
- Four FLA mixers and four recurrent schedules: finite gradients.
- Checkpoint round-trip: exact logits.
- Package wheel: built and imported from an isolated target directory, including
  the bundled `utilis` package.
- Optimized ASV19 test entrypoint: model reload and prediction-table smoke passed;
  outputs were written under `tested_results`.
- Optimized ASV19 and ASV21 score entrypoints: complete prediction tables were
  scored and summaries were written under `scored_results`.
- Legacy model/loss/utility import search: no matches; the bundled RawBoost resolves
  through `code_optimize/utilis/rawboost_utils.py` before collation.

## Expected dependency notices

The installed FLA/Triton stack emits deprecation notices for
`torch.get_autocast_gpu_dtype()` and `tl.make_block_ptr`. Mamba3 has a visible
first-run Triton compilation delay. These are documented upstream-bound
observations, not hidden or suppressed by this implementation.
