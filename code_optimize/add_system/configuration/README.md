# Configuration

This directory turns YAML experiment files into validated Hugging Face config
objects and runtime settings. It is the boundary between editable experiment
descriptions and Python model construction.

## Files and dependencies

```text
configuration/
├── __init__.py       public configuration API
├── validation.py     scalar and nested-config validation helpers
├── blocks.py         block enums, BlockSpec, normalization, and expansion
├── components/
│   ├── add.py        top-level ADDConfig and pipeline contracts
│   ├── aasist.py     AASIST component parameters
│   ├── backbone.py   backbone dimensions, schedules, and block groups
│   ├── classifier.py classifier head parameters and type
│   ├── frontends.py  linear, SincNet, and SSL frontend parameters
│   ├── mixers.py     Attention, Raven, GDN2, and Mamba3 parameters
│   └── pooling.py    pooling-family parameters and pooling type
├── experiments.py    strict YAML loading, merge, digest, and resolution
├── factory.py        YAML mappings -> validated ADDConfig
└── settings.py       resolved YAML -> run/data/loss/train settings
```

The dependency flow is:

```text
configs/common.yaml + experiment.yaml
                    |
                    v
             experiments.py
       load_yaml_mapping / deep_merge
                    |
                    v
             ResolvedExperiment
                /       \
               v         v
          factory.py   settings.py
               |            |
               v            v
           ADDConfig  ExperimentSettings
               |
               v
       add_system.models
```

`validation.py` is used by both `components/` and `settings.py`. It rejects
wrong types, invalid ranges, unknown nested fields, and incompatible model
types. `experiments.py` also rejects duplicate YAML keys before merging.

## Component configs

`components/add.py` is the top-level contract. It validates that the selected
pipeline has the right children and that adjacent dimensions match:

```text
ADDConfig
├── frontend_config
├── backbone_config       required by backbone_pooling
├── pooling_config        required by backbone_pooling
├── aasist_config         required by ssl_aasist
└── classifier_config
```

`factory.py` performs the schema check before constructing `ADDConfig`, so a
YAML typo is reported before importing or instantiating the model modules.
`settings.py` handles the non-model sections:

```text
ExperimentSettings
├── run          task name, output root, seed, GPU selection
├── data         cache path, split names, columns, sample limits
├── loss         cross-entropy/focal settings and class weights
├── augmentation RawBoost settings
└── training     Trainer, optimizer, schedule, checkpoint settings
```

## Block specification flow

The `blocks` entries describe repeated mixer/feed-forward groups. They are
normalized once into immutable `BlockSpec` objects:

```text
YAML
└── blocks:
    ├── attn: raven | gdn2 | mamba3 | attention
    ├── mlp:  mlp | moe
    └── num_layers: N
          |
          v
normalize_block_specs(...)
          |
          v
tuple[BlockSpec, ...]
          |
          v
expand_block_specs(...)
          |
          v
one BlockSpec per physical layer
```

`AttentionBackboneConfig.block_specs` exposes the normalized main stack.
Heterogeneous HRM uses `hrm_h_block_specs` and `hrm_l_block_specs` for separate
H and L stacks. During model construction, the architecture factory creates a
stack; `BackboneStack` expands the specs and creates one
`SequenceFeedForwardBlock` per physical layer. Each sequence block then calls
the model factories:

```text
Architecture
    -> BackboneStack
        -> SequenceFeedForwardBlock
            -> mixers.factory.build_mixer
            -> feedforward.factory.build_feed_forward
```

The same normalized block schema can therefore describe baseline, looped, HRM,
and heterogeneous HRM models without duplicating parser logic in the model
code. `blocks.py` describes topology; `components/backbone.py` supplies shared
dimensions and mixer-specific config objects; model files perform execution.

## Adding configuration for a module

Configuration work has three separate responsibilities:

```text
serialized name and topology -> blocks.py or a component enum
module parameters            -> components/*.py
YAML mapping validation      -> factory.py
```

Keep these responsibilities separate from model construction. Config classes
validate types, ranges, compatible dimensions, and legal component
combinations; model factories receive already validated objects and only
construct modules.

### Reusing an existing component config

When an implementation shares the same parameter schema, add a new enum value
and register the implementation in its model factory. Pooling methods are the
current example: all use `GatedAttentionPoolingConfig`, while `PoolingType`
selects the concrete module. No new top-level config field is needed.

### Adding a new config type

When parameters or serialization differ:

1. Create a `PretrainedConfig` subclass in the appropriate `components/` file
   with a stable, unique `model_type`.
2. Validate constructor values using `validation.py`; store JSON-compatible
   primitives on the config object.
3. Add the type to the owning union/coercion path. Frontends use
   `FRONTEND_CONFIG_TYPES` in `factory.py` and `_coerce_frontend` in
   `components/add.py`; mixer configs are nested in
   `AttentionBackboneConfig`.
4. Add nested-field validation in `build_add_config` so unknown YAML keys fail
   before model construction. A mixer config must be added to the
   `(field_name, config_type)` validation table, not only to the model factory.
5. Export the config from `components/__init__.py` when it is part of the public
   construction API.

`model_type` identifies serialized config classes. An implementation selector
such as `pooling_type`, `MixerType`, or `ArchitectureType` chooses behavior
inside an owning config; the two concepts should not be used interchangeably.

### Adding topology

A new mixer or feed-forward name extends `MixerType` or `FeedForwardType` and
therefore becomes legal in `BlockSpec`. A new execution schedule extends
`ArchitectureType` and requires matching schedule validation in
`AttentionBackboneConfig`. A new end-to-end branch extends `PipelineType` and
requires an explicit component contract in `ADDConfig`.

For every top-level pipeline, `ADDConfig` must state:

- which nested configs are required or forbidden;
- every adjacent input/output dimension equality;
- the resolved architecture name saved in checkpoints;
- any runtime feature restrictions such as cache or attention support.

### YAML overlay rules

Add shared defaults to `configs/common.yaml` only when all experiments should
inherit them. Put the new model choice in a focused overlay under
`configs/experiments/`. Nested dictionaries merge recursively, except that a
mapping with a different `model_type` replaces the previous component mapping
instead of retaining incompatible fields.

After wiring a config, verify both direct construction and YAML construction.
The catalog test currently compares both the semantics and the exact set of
experiment paths against the migrated legacy catalog. Therefore a new file
under `configs/experiments/` must also exist in the migration source unless the
catalog policy and test are intentionally changed. Optimized-only smoke profiles
can remain directly under `configs/`.
