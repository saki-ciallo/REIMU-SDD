# Tests

The tests cover the optimized package boundary. They are grouped by behavior
rather than by one-to-one correspondence with implementation directories.

## Tree

```text
tests/
├── test_config_catalog.py       legacy-to-new config catalog parity
├── test_config_schema.py        block and schema validation
├── test_experiment_config.py    YAML merge and resolved settings
├── test_model_config.py         component dimension contracts
├── test_components.py           frontend/backbone/pooling/classifier units
├── test_data.py                 dataset cache and collation behavior
├── test_training.py             losses and Trainer contracts
├── test_cuda_backbones.py       FLA mixer gradients
├── test_cuda_moe.py             CUDA MoE grouped-matrix paths
├── test_cuda_training_smoke.py  minimal CUDA training path
├── test_diagnostics.py          reports, manifests, and benchmark helpers
└── test_package_layout.py       self-contained package and entry points
```

The normal command from the repository root is:

```bash
PYTHONPATH=code_optimize python -m unittest discover -s code_optimize/tests -v
```

CUDA tests intentionally require the configured GPU and validate finite
forward/backward behavior for the actual FLA implementations. They are not
CPU fallback tests.
