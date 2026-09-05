# Models

This directory contains the trainable computation graph. `ADDModel` is the
only top-level model assembled by the training runner; the other classes are
component wrappers or reusable layers selected by configuration.

## Model tree

```text
models/
├── __init__.py       public model classes and output types
├── add.py            ADDModel: frontend -> backend -> classifier
├── outputs.py        Hugging Face ModelOutput dataclasses
├── frontends/
│   ├── factory.py    selects linear, SincNet, or SSL frontend
│   ├── linear.py     fixed frame slicing and projection
│   ├── sincnet.py    SincConv plus convolutional encoder
│   └── ssl.py        pretrained SSL encoder and optional projection
├── backbones/
│   ├── backbone.py   HF wrapper and FLA initialization
│   ├── backbone_stack.py one complete stack and post-norm
│   ├── architectures/ baseline, looped, HRM, heterogeneous HRM
│   ├── blocks/       one mixer plus feed-forward residual block
│   ├── mixers/       Attention, Raven, GDN2, and Mamba3 factory
│   └── feedforward/  GatedMLP, router, and LatentMoE
├── pooling/
│   ├── pooling.py    HF wrapper and output contract
│   ├── factory.py    selects the configured pooling family
│   ├── mhgap.py      multi-head gated attention pooling
│   ├── gamp.py       global average/max pooling
│   ├── gapv1.py      gated attention pooling V1
│   └── gapv2.py      gated attention pooling V2
├── classifier/
│   ├── classifier.py HF wrapper
│   ├── head.py       linear or AMSoftmax projection
│   └── amsoftmax.py  margin-logit calculation
└── aasist/
    ├── aasist.py     HF AASIST wrapper
    ├── encoder.py    temporal/spectral encoder paths
    └── graph.py      graph attention and graph pooling layers
```

## Top-level assembly

`ADDModel.__init__` follows the validated `ADDConfig`:

```text
ADDConfig
   |
   +--> build_frontend(frontend_config) -> frontend
   |
   +--> backbone_pooling
   |       +--> BackboneModel(backbone_config)
   |       +--> PoolingModel(pooling_config)
   |
   +--> ssl_aasist
           +--> AASISTModel(aasist_config)

both branches -> ClassifierModel(classifier_config)
```

The normal `backbone_pooling` forward path is:

```text
input_values [B, samples]
        |
        v
frontend.hidden_state [B, sequence, hidden_size]
        |
        v
BackboneModel.last_hidden_state [B, sequence, hidden_size]
        |
        v
PoolingModel.pooled_output [B, classifier_input_size]
        |
        v
ClassifierModel.logits [B, num_labels]
```

The `ssl_aasist` branch skips backbone and pooling:

```text
input_values -> SSLFrontendModel -> AASISTModel -> ClassifierModel
```

`outputs.py` carries optional hidden states, attentions, cache, auxiliary MoE
loss, and AASIST intermediate states without making the classifier responsible
for those details. The training loss is deliberately computed by
`training/ADDTrainer`, outside the model.

## Backbone dependency flow

```text
BackboneModel
    -> architectures.factory.build_architecture
        -> BaselineArchitecture
        -> LoopedArchitecture
        -> HRMArchitecture
        -> HeterogeneousHRMArchitecture
              |
              v
        BackboneStack
              |
              +--> expand_block_specs
              |
              v
        SequenceFeedForwardBlock (one per physical layer)
              |
              +--> attn_norm -> mixers.factory.build_mixer -> residual
              |
              +--> mlp_norm  -> feedforward.factory.build_feed_forward
              |                 -> GatedMLP or LatentMoE -> residual
              v
        optional recurrent post_norm
```

`BackboneStack` is the reusable `(blocks + optional post-norm)` module. The
baseline executes it once without a terminal post-norm. Looped reuses one
stack for multiple cycles and enables gradients only in the configured final
cycles. HRM owns separate shared H and L stacks and schedules their repeated
calls; heterogeneous HRM supplies independent H and L block specifications.

`SequenceFeedForwardBlock` owns the pre-norm residual ordering and receives
one `BlockSpec`. It does not decide which mixer or feed-forward type to use:
the two small factories do that from the spec and backbone config. This keeps
the block implementation independent from the YAML parser.

## Frontend, pooling, and classifier factories

The component factories keep selection at construction time:

```text
frontend config -> frontends.factory.build_frontend
pooling config  -> pooling.factory.build_pooling
block spec      -> mixers.factory.build_mixer
block spec      -> feedforward.factory.build_feed_forward
classifier config -> ClassifierHead
```

The SSL frontend can be frozen, fully fine-tuned, or fine-tuned on selected
encoder layers. If the loaded SSL width differs from the backbone width, its
output projection supplies the configured dimension explicitly. Pooling owns
its activation, optional output normalization, and dropout. The classifier
returns ordinary logits; AMSoftmax only supplies margin-adjusted logits for
the external training loss when selected.

## Extending the model

Choose the smallest extension point that represents the new behavior. A new
pooling method belongs in the pooling registry; a new recurrence schedule is a
backbone architecture; only a genuinely different end-to-end route belongs in
`ADDModel`. This keeps component choices out of forward methods and keeps the
training runner independent of model internals.

### Component contracts

Every implementation must preserve the contract used by its caller:

| Component | Input | Output expected by the caller |
| --- | --- | --- |
| frontend wrapper | waveform `[B, samples]` | `FrontendOutput.hidden_state` `[B, S, H]` |
| mixer | hidden state plus mask/cache flags | `(hidden_states, attention, past_key_values)` |
| dense feed-forward | hidden state `[B, S, H]` | tensor `[B, S, H]` |
| MoE feed-forward | hidden state `[B, S, H]` | `MoEOutput(hidden_states, aux_loss)` |
| pooling implementation | hidden state `[B, S, H]` | pooled tensor `[B, P]` |
| classifier wrapper | pooled state `[B, P]` | `ClassifierOutput` with prediction logits |
| backbone architecture | hidden state and runtime flags | `ArchitectureOutput` |

Dimension changes must be explicit in the owning module and validated by its
configuration. Do not add silent projection fallbacks between components.
Parameter initialization should remain in the nearest Hugging Face wrapper's
`_init_weights`/`post_init` path when that wrapper already owns initialization.

### Adding a leaf module

For a new frontend:

1. Add its `PretrainedConfig` in `configuration/components/frontends.py`,
   including a unique `model_type` and `frontend_output_dim`.
2. Implement the wrapper under `models/frontends/` and return
   `FrontendOutput`.
3. Add the config/model pair to `models/frontends/factory.py` and add the
   config type to the configuration factory and top-level frontend coercion.
4. Export public classes only when callers need to construct them directly.

For a new pooling method that can share `GatedAttentionPoolingConfig`:

1. Add a value to `PoolingType` in `configuration/components/pooling.py`.
2. Implement one `nn.Module` under `models/pooling/`; keep optional activation,
   output norm, and dropout consistent with the shared pooling contract.
3. Register the class in `POOLING_TYPES` in `models/pooling/factory.py`.
4. Extend `PoolingModel._init_weights` only when the module introduces a
   parameter that the existing linear/norm initialization does not cover.

For a new backbone mixer:

1. Add its serialized name to `MixerType` in `configuration/blocks.py`.
2. Define its typed parameters in `configuration/components/mixers.py`, then
   attach that config to `AttentionBackboneConfig` and validate dimensions only
   when the mixer is selected.
3. Add the nested config to the mixer-validation table in
   `configuration/factory.py`, then add one construction branch to
   `models/backbones/mixers/factory.py`.
4. Make the mixer return the three-value FLA contract; cache and attention
   placeholders must keep their positions even when unsupported.

For a new feed-forward implementation, add its name to `FeedForwardType`, add
its parameters to the backbone config, and construct it in
`models/backbones/feedforward/factory.py`. Return a tensor for a dense module or
`MoEOutput` when an auxiliary loss must be propagated. The residual and norm
ordering remains owned by `SequenceFeedForwardBlock`. A new auxiliary-loss
module must also update `_forward_feed_forward` there: its current collection
policy intentionally recognizes `LatentMoE`, so a factory branch alone does not
make a different MoE class contribute to `ADDOutput.aux_loss`.

For a new classifier head:

1. Add its serialized selector and parameters in
   `configuration/components/classifier.py`.
2. Implement the projection under `models/classifier/` and select it in
   `models/classifier/head.py`.
3. Keep label-independent prediction logits separate from any label-dependent
   training logits. Extend `ClassifierOutput`, `ADDOutput`, and the trainer loss
   path only when the new head requires a genuinely new output.
4. Keep `ClassifierModel` responsible for the Hugging Face wrapper contract and
   centralized initialization.

### Adding a backbone architecture

A backbone architecture controls repeated calls, state flow, post-normalization,
and gradient scheduling; it does not rebuild individual layers. To add one:

1. Add a value to `ArchitectureType` in `configuration/blocks.py` and validate
   its schedule fields in `AttentionBackboneConfig`.
2. Implement a `BackboneArchitecture` subclass under
   `models/backbones/architectures/`, reusing `BackboneStack` for each complete
   parameterized stack.
3. Register the class in `models/backbones/architectures/factory.py`.
4. Return `ArchitectureOutput` and preserve hidden-state, cache, attention, and
   auxiliary-loss collection semantics.

Looped or hierarchical execution must make parameter sharing explicit. A
`torch.no_grad` or `torch.set_grad_enabled` region controls retained activation
history for a call; it must not silently change parameter `requires_grad`.

### Adding a complete pipeline

Use this path only when the sequence of major components differs from both
`backbone_pooling` and `ssl_aasist`:

1. Add a `PipelineType` value and any new component config class.
2. Extend `ADDConfig` with the required/forbidden component combination and all
   adjacent dimension checks.
3. Teach `configuration/factory.py` to validate and build the nested config.
4. Construct the branch in `ADDModel.__init__` and execute it in
   `ADDModel.forward`.
5. Add any genuinely new public state to `outputs.py`; do not overload an
   existing output field with a different meaning.

If the branch requires a different batch input, update `data/collation.py` and
the model's public forward signature together. Direct
`ADDModel.from_pretrained(...)` uses the serialized `ADDConfig`; if the model
must also load through Hugging Face `AutoModel`, keep the explicit registration
in `add_system/registration.py` aligned.

Loss selection, class weights, datasets, and checkpoint policy remain in
`training/` and should not be moved into the new model branch.

### Selecting registered modules

Once the typed config and model factory know a leaf module, experiment YAML can
select it without another branch in `ADDModel`. For example, registered mixer
and pooling names can be composed in an overlay as follows:

```yaml
model:
  backbone_config:
    blocks:
      - attn: my_mixer
        mlp: mlp
        num_layers: 6
  pooling_config:
    model_type: gap_family
    pooling_type: my_pooling
```

`resolve_experiment(...).build_settings()` validates the merged YAML and
produces the `ADDConfig` passed to `ADDModel`. This is the normal construction
route for training, testing, checkpoint restoration, and diagnostics.

### Verification checklist

After adding a module:

1. Add a minimal YAML overlay under `configs/experiments/`.
2. Test validation, serialization, and dimension mismatches in
   `tests/test_model_config.py` or `tests/test_experiment_config.py`.
3. Test shape, dtype, and backward behavior in `tests/test_components.py` or
   the relevant CUDA backbone test.
4. Run `tests/test_cuda_training_smoke.py` when assembly or checkpoint loading
   changes.
5. Confirm `model_overview.md` names the concrete component; update diagnostics
   only if the new type is not already discovered generically.

`tests/test_config_catalog.py` currently requires the optimized experiment tree
to mirror the legacy catalog exactly. When adding an optimized-only overlay
under `configs/experiments/`, either add its migration source as well or make a
deliberate change to that catalog policy and its test.

Run the optimized suite from the repository root:

```bash
PYTHONPATH=code_optimize python -m unittest discover -s code_optimize/tests -v
```
