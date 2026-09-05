# Smoke-test and troubleshooting log

Every entry records the symptom, attempted diagnosis, root cause, final action,
and verification command.

## 2026-07-31: `pytest` was unavailable in `py312`

**Symptom**

Both `pytest` and `python -m pytest` failed because the module was not installed.

**Options considered**

1. install pytest directly into the shared environment;
2. declare pytest as an optional project dependency;
3. use a standardized development dependency group;
4. use Python's built-in `unittest` runner.

**Final action**

The initial configuration tests use `unittest`, keeping the independent
implementation runnable without mutating the existing environment. A dedicated
test dependency group can be introduced if later fixtures or parameterization
justify pytest.

**Verification**

```bash
python -m unittest discover -s tests -v
```

## 2026-07-31: Ruff repeatedly reported `I001`

**Symptom**

Manual import edits still produced `unsorted-imports`.

**Final action**

Used Ruff's documented two-stage workflow:

```bash
ruff check --no-cache --select I --fix .
ruff format --no-cache .
```

Disabling `I001`, adding `noqa`, and maintaining a separate isort configuration
were rejected because they weaken or duplicate the shared style contract.

## 2026-07-31: `warmup_steps: 0.05` looked like an invalid integer

**Symptom**

A two-step Trainer smoke logged learning rates `0` and `1e-4`, suggesting that
the common YAML might be passing a ratio through an integer field.

**Investigation**

1. Parsed the merged YAML and confirmed `warmup_steps` was a float.
2. Temporarily tried `warmup_ratio: 0.05`.
3. Transformers 5.14 emitted a deprecation warning for `warmup_ratio`.
4. Inspected the installed `TrainingArguments` field and
   `get_warmup_steps()` implementation.

**Root cause**

Transformers 5.14 changed `warmup_steps` to accept either an absolute integer
or a float ratio in `[0, 1)`. For a two-step smoke, `ceil(2 * 0.05) = 1`, so
the observed schedule was expected.

**Final action**

Restored `warmup_steps: 0.05`. No reference code change was retained.

**Verification**

```python
assert training_args.get_warmup_steps(1000) == 50
```

## 2026-07-31: Dynamo warned about an `lru_cache`-wrapped FSDP helper

**Symptom**

The first compiled Trainer step warned that Dynamo directly traces
`transformers.distributed.fsdp.is_fsdp_managed_module` instead of its
`functools.lru_cache` wrapper.

**Investigation**

Ran two training steps followed by an independent evaluation using a frozen
Wav2Vec2 frontend and Attention backbone.

**Result**

- compiled train loss: `0.06930`;
- compiled eval loss: `0.06921`;
- finite gradient norms: `0.00304`, `0.00237`;
- checkpoint save completed;
- no lingering GPU process remained.

**Final action**

Treat the warning as informational for this dependency combination. Do not
suppress it globally; revisit only if numerical or graph-break evidence appears.

## 2026-07-31: baseline gradient smoke

Selected Wav2Vec2 layers 10 and 11, the Attention backbone, MHGAP, and the
linear classifier all received finite nonzero gradients under BF16 autocast.
Two manual AdamW steps and two real Trainer steps completed successfully.

## 2026-07-31: tests depended on the current working directory

**Symptom**

The full suite passed when launched inside `code_optimize/` but failed from the
repository root with:

```text
FileNotFoundError: .../local-dev-ADD/configs/common.yaml
```

**Root cause**

One test used `"configs/common.yaml"` relative to the shell working directory.
The implementation was stable; the test fixture was not location-independent.

**Final action**

Resolve fixture paths from `Path(__file__).resolve().parents[1]`. The suite now
passes from the repository root, which is the documented invocation.

## 2026-07-31: FLA checkpoint smoke failed on CPU and FP32

**Attempts**

1. CPU forward failed because fused FLA normalization is CUDA-only.
2. CUDA FP32 forward reached FlashAttention, which requires FP16 or BF16.
3. CUDA with BF16 autocast completed.

**Final action**

Keep the stated CUDA-only contract and test checkpoint round-trips under BF16
autocast. Saving and restoring the same model produced exactly equal logits
(`rtol=0`, `atol=0`).

No CPU fallback or silent dtype conversion was added.

## 2026-07-31: cache inspection used a private field

**Symptom**

The continuation forward succeeded, but the diagnostic script raised:

```text
AttributeError: 'Cache' object has no attribute 'seen_tokens'
```

**Root cause**

Current FLA exposes `Cache.get_seq_length()` rather than a public
`seen_tokens` attribute.

**Final action**

Use the public method. A two-layer baseline processed 8 tokens, reused the
returned cache for 4 more, and reported sequence length 12 with finite outputs.

## 2026-07-31: Mamba3 smoke appeared to stall

**Symptom**

The first Mamba3 backward smoke spent roughly one minute without new test
output.

**Root cause**

The first execution compiled Triton kernels. Subsequent runs reused the cache
and completed quickly.

**Observed upstream warnings**

- Triton warns that `tl.make_block_ptr` is deprecated.
- FLA normalization warns that `torch.get_autocast_gpu_dtype()` is deprecated.

Both warnings originate in installed dependencies. They are not suppressed
because doing so could hide a future compatibility break. The four FLA mixers
all produced finite gradients after compilation.

During the compatibility audit, switching model defaults to the older
`config.use_return_dict` convention produced a Transformers 5.14 deprecation
warning. The implementation uses `config.return_dict`, which is the supported
field in the installed version.

The same audit showed that Transformers 5.14 removed
`TrainingArguments.overwrite_output_dir`. The field was not retained in the new
schema. Checkpoint continuation uses the still-supported
`Trainer.train(resume_from_checkpoint=...)` interface.

## 2026-07-31: real Trainer and compile verification

The following paths were run against four cached ASVspoof2019 examples:

| Mode | Train steps | Train loss | Eval loss | Result |
|---|---:|---:|---:|---|
| eager | 2 | 0.6931 | 0.6931 | passed |
| `torch.compile`, `default` | 2 | 0.6931 | 0.6931 | passed |

The compiled smoke had a cold train runtime of approximately 3.9 seconds. The
final evaluation completed normally. The runner does not mutate private Dynamo
configuration such as `capture_dynamic_output_shape_ops`, avoiding the
cross-version failures seen with PyTorch 2.9 and 2.12.

## 2026-07-31: end-to-end optimizer evidence

For a linear frontend, one Attention/GatedMLP layer, MHGAP, and a linear
classifier under BF16 autocast:

- logits shape and dtype: `[2, 2]`, BF16;
- weighted cross-entropy: `0.6931065`, finite;
- gradient tensors: 15;
- global gradient norm: `0.0083724`, finite and nonzero;
- maximum tracked AdamW parameter update: `0.0010006`.

This verifies more than gradient presence: one actual fused AdamW step changed
trainable parameters.

## 2026-07-31: retired CUDA RawBoost experiment

The CUDA FFT implementation was originally compared with the canonical
NumPy/SciPy implementation using identical sampled parameters:

- maximum absolute difference: `1.49e-8`;
- mean absolute difference: `7.80e-11`;
- output shape and FP32 dtype preserved;
- all values finite.

It is no longer used by either training entrypoint. All RawBoost algorithms,
including algorithm 4, now run through the bundled
`code_optimize/utilis/RawBoost.py` as a CPU Dataset transform before collation.
This keeps augmentation outside `ADDModel`, so model
FLOP and inference-latency accounting exclude preprocessing work.

The LatentMoE regression smoke also completed with native BF16
`torch.nn.functional.grouped_mm`: router and expert gradients were finite and
nonzero, and the sequence-level auxiliary loss was finite and positive.

## 2026-07-31: wheel build command treated a path as a package name

**Symptom**

`pip wheel ... code_optimize` searched the package index for a distribution
named `code_optimize`.

**Final action**

Use an explicit filesystem path:

```bash
python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /tmp/add_system_wheels ./code_optimize
```

The resulting `add_system-0.1.0-py3-none-any.whl` built successfully.

## 2026-07-31: AMSoftmax evaluation logits used the labels

**Symptom**

The first AMSoftmax interface returned margin-adjusted logits whenever labels
were passed. Hugging Face evaluation passes labels through `compute_loss`, so
accuracy could be computed from logits that already depended on the true class.

**Root cause**

Prediction scores and loss-specific margin scores shared one output field.

**Final action**

The classifier now always returns label-independent `logits` and separately
returns `loss_logits` for AMSoftmax when labels are supplied. `ADDTrainer` uses
`loss_logits` only for loss computation; metrics and prediction tables continue
to consume `logits`.

A regression test confirms that prediction logits are identical with and
without labels and that the target class receives exactly
`margin * scale` in `loss_logits`.

A real two-step AMSoftmax Trainer smoke then completed with finite train/eval
losses (`4.145` and approximately `3.18`) and `0.75` accuracy on the four-sample
fixture. No NaN was observed.

## 2026-07-31: four SSL frontend smoke

With locally cached weights, a one-second waveform and BF16 CUDA autocast:

| SSL frontend | Output | Dtype | Finite | Mode |
|---|---|---|---|---|
| Wav2Vec2 Base | `[1, 49, 128]` | BF16 | yes | frozen |
| HuBERT Base LS960 | `[1, 49, 128]` | BF16 | yes | frozen |
| WavLM Base | `[1, 49, 128]` | BF16 | yes | frozen |
| WavLM Base Plus | `[1, 49, 128]` | BF16 | yes | frozen |

Wav2Vec2 reports quantizer and pretraining projection keys as unexpected when
loading the base encoder. This is expected because `Wav2Vec2Model` omits the
pretraining-only heads.

HuBERT emitted a one-time unauthenticated-Hub notice during the sequential
smoke. A separate process with both `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` loaded the cached model successfully, confirming that
`local_files_only=True` does not require network access.

Selected-layer tuning was also checked with Wav2Vec2 layers 10 and 11 plus the
768-to-128 output projection. All 33 trainable tensors received finite nonzero
gradients under BF16 autocast; frozen SSL parameters remained excluded.

## 2026-07-31: inference benchmark smoke

The Batch-1 benchmark completed in eager and compiled modes and wrote valid
JSON. For the tiny one-layer linear-frontend fixture on the RTX 4080 SUPER:

| Mode | Median latency | RTF | Real-time speedup |
|---|---:|---:|---:|
| eager | 1.22 ms | 0.00122 | 820.6x |
| compiled | 1.99 ms | 0.00199 | 501.5x |

These numbers only validate the tool. They also demonstrate why compile should
be measured rather than assumed faster: launch overhead dominates this tiny
model. Production comparisons must use the same checkpoint, audio duration,
warmup, measurement count, and hardware.

`max-autotune-no-cudagraphs` also completed. During search, Inductor logged
several rejected Triton candidates as `No valid triton configs` because their
shared-memory requirements exceeded the GPU limit. Autotuning continued,
selected a valid kernel, and produced finite benchmark output. These candidate
rejections are noisy search diagnostics, not process-level CUDA OOM failures.

## 2026-07-31: FLA Attention cannot return attention maps

FLA 0.5.2 declares `output_attentions` in `Attention.forward`, but the current
implementation only assigns its local `attentions` variable when the flag is
false. Passing `true` would therefore raise an unbound-local error rather than
return a map.

The optimized stack calls upstream mixers with `output_attentions=False` and,
when the top-level option is requested, returns one `None` slot per layer. This
preserves output ordering for mixed Attention/Raven/GDN2/Mamba3 stacks without
pretending that maps exist. Revisit this only after the installed FLA version
actually materializes attention weights.
