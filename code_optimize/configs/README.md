# Experiment Configurations

This directory contains the executable YAML catalog for `code_optimize`. YAML
describes a run; Python code in `add_system/configuration/` validates and
resolves it before the model is built.

## Tree

```text
configs/
├── common.yaml                 shared model/data/training defaults
├── smoke.yaml                  short non-compile validation profile
├── smoke_compile.yaml          short Inductor validation profile
├── smoke_pipeline.yaml         pipeline-specific smoke overrides
└── experiments/
    ├── baseline/               one backbone stack
    ├── finetune/               SSL selected-layer and RawBoost runs
    ├── finetune_Top1/          Top-1 SSL layer variants
    ├── finetune_Top3/          Top-3 SSL layer variants
    ├── hetero/                 heterogeneous HRM H/L stacks
    ├── hrm/                    homogeneous HRM stacks
    └── looped/                 shared recurrent stack runs
```

## Resolution flow

```text
common.yaml + one experiment overlay
                  |
                  v
configuration.experiments.resolve_experiment
                  |
                  +--> strict YAML parsing and duplicate-key checks
                  +--> recursive mapping merge
                  +--> ResolvedExperiment digest
                  |
                  v
configuration.factory / settings
                  |
                  v
             ADDConfig + ExperimentSettings
```

`common.yaml` supplies defaults and an experiment file overrides only the
values relevant to that run. Model sections use explicit nested component
configs, including `frontend_config`, `backbone_config`, `pooling_config`,
and `classifier_config`. Training, data, loss, augmentation, and run settings
are separate sections.

The normal launcher passes the common file and experiment file in order. The
maintenance script [`migrate_configs_to_new.py`](../migrate_configs_to_new.py)
converts the legacy root catalog when the source configurations change.
