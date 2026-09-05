# Design decisions

## Prefer explicit types at configuration boundaries

External YAML remains JSON-compatible, but it is normalized immediately into
`StrEnum` values and immutable `BlockSpec` instances. Models should not repeat
string normalization or validate arbitrary dictionaries.

## Use modern Python only where it improves clarity

The optimized package targets Python 3.12 and uses:

- `StrEnum` for serialized option sets;
- `slots=True` and `frozen=True` for immutable value objects;
- PEP 695 `type` aliases and generic function syntax;
- `zip(..., strict=True)` where equal lengths are an invariant;
- structural pattern matching only when it is clearer than direct branching.

Loops and `if` statements are retained when they express model schedules most
clearly. They are not replaced merely for novelty. Python `assert` is not used
for user/configuration validation because optimized interpreter mode can remove
assertions.

## Keep Hugging Face configuration serialization

Model configs continue to derive from `PretrainedConfig`. Typed helper objects
are serialized back to primitive dictionaries before being stored on HF config
instances, preserving JSON checkpoint compatibility.

## Registry scope

Registries are appropriate for leaf module selection, such as mixer, pooling,
or classifier type. Architecture schedules remain explicit classes because
their control flow and gradient semantics are materially different.

Hugging Face `AutoConfig`/`AutoModel` registration is exposed through an
explicit function. Importing `add_system` does not mutate global AutoClass
registries.

## External model loading

SSL configuration includes `local_files_only`. It defaults to `false` so a new
machine can perform the initial download; experiments can set it to `true`
after caching to remove network probes and make startup behavior deterministic.
Checkpoint restoration embeds the upstream SSL configuration so meta-device
construction does not call `from_pretrained`.

## CUDA and dtype contract

Training settings require BF16. FLA attention and the native grouped-MoE
weights do not provide a supported FP32/CPU fallback in this implementation.
Invalid precision is rejected at configuration time instead of failing inside
a later kernel call.

Transformers 5.14 deprecates `PretrainedConfig.use_return_dict` in favor of the
direct `return_dict` field. Public forward methods therefore resolve
`return_dict=None` from `config.return_dict`.

## Performance policy

Optimization work follows measurements:

1. preserve numerical behavior;
2. measure eager and compile modes independently;
3. prefer fused/upstream kernels already provided by PyTorch or FLA;
4. avoid Python-level dispatch inside token/layer hot paths;
5. document compilation cost separately from steady-state speed.

Inference benchmark timing uses CUDA events after explicit warmup. Checkpoint
loading, SSL download, and first-kernel compilation are reported separately
from steady Batch-1 latency and RTF.

Trainer inference ignores pooled states, cache/history, auxiliary loss, and
AMSoftmax loss-specific logits. Only label-independent class logits are
accumulated for metrics, avoiding validation-memory growth from diagnostic
outputs.
