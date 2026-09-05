# Bundled Utilities

These helpers are shared by the optimized training and evaluation entry points
but do not define model layers.

## Tree

```text
utilis/
├── RawBoost.py              original CPU/SciPy RawBoost implementation
├── rawboost_utils.py        typed adapter and Dataset transform builder
├── model_test_utils.py      checkpoint, dataset, and prediction helpers
├── model_resutls_utils.py   tested/scored result directory helpers
└── asv19_metrics.py         local ASVspoof2019 metric functions
```

`RawBoost.py` is intentionally kept algorithmically unchanged and is excluded
from the optimized formatter/linter rules. `rawboost_utils.py` connects it to
`add_system/data/preprocessing.py`; augmentation therefore stays outside the
model graph.

`model_test_utils.py` is used by `evaluation/*_test.py` for checkpoint
discovery and prediction generation. `model_resutls_utils.py` centralizes the
`tested_results` and `scored_results` paths. `asv19_metrics.py` supplies local
ASVspoof2019 metrics, while the dataset-specific workflow remains under
`evaluation/`.
