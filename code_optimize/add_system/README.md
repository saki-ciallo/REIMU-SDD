# `add_system`

`add_system` is the self-contained Python package for the optimized ADD
implementation. It owns typed configuration, data preparation, model
construction, training, and diagnostics. The shell entry points in the parent
directory call into this package through `PYTHONPATH`.

## Package tree

```text
add_system/
├── configuration/  typed configs, YAML resolution, and validation
├── data/           dataset loading, transforms, and collation
├── diagnostics/    model reports, run manifests, and benchmarks
├── models/         frontend, backbone, pooling, AASIST, and classifier
└── training/       Trainer adapter, losses, arguments, and runner
```

Each directory has its own README for the files it owns. The package-level
dependency direction is:

```text
configs/*.yaml
      |
      v
configuration/experiments.py
      |------------------------------+
      v                              v
configuration/factory.py       configuration/settings.py
      |                              |
      v                              v
  ADDConfig                 ExperimentSettings
      |                              |
      +--------------+---------------+
                     v
              training/run.py
                /       \
               v         v
          data/         models/
               \         /
                v       v
             diagnostics/
```

The actual model path is selected by `ADDConfig.model_architecture`:

```text
input_values
    |
    v
frontend
    |
    +--> backbone -> pooling --+
    |                           |
    +--> AASIST ----------------+
                                v
                            classifier
                                |
                                v
                              logits
```

`backbone_pooling` uses a configured frontend, `BackboneModel`,
`PoolingModel`, and `ClassifierModel`. `ssl_aasist` uses an SSL frontend,
`AASISTModel`, and the same classifier contract. Both paths return the shared
output structures from `models/outputs.py`.

## Runtime ownership

- `configuration/` validates choices and dimensions before model construction.
- `data/` converts cached Hugging Face examples into model-ready batches; its
  optional RawBoost transform remains outside the model graph.
- `models/` performs all forward computation and owns model parameters.
- `training/` builds the Hugging Face `TrainingArguments`, loss, and
  `ADDTrainer`, then coordinates data and model objects.
- `diagnostics/` reads the assembled model without changing its computation;
  it writes reports, manifests, and optional inference measurements.

The intended import path for the normal workflow is therefore:

```text
YAML -> configuration -> training/run.py -> data + models -> diagnostics
```

## Extension flow

New behavior should enter through the narrowest stable boundary:

```text
serialized option
    -> typed component config and validation
    -> one model implementation
    -> the owning factory or registry
    -> experiment YAML
    -> component and assembly tests
```

A leaf component such as a frontend, mixer, feed-forward layer, pooling method,
or classifier should not add branches to the training runner. A new backbone
schedule extends the architecture layer while reusing existing blocks. Only a
new end-to-end component order extends `ADDConfig` and `ADDModel` as a complete
pipeline.

See [models/README.md](models/README.md#extending-the-model) for runtime
contracts and exact construction points. See
[configuration/README.md](configuration/README.md#adding-configuration-for-a-module)
for enums, typed configs, YAML validation, dimension checks, and serialization.
