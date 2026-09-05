# Evaluation

This directory keeps dataset-specific inference and official scoring separate
from training. The parent `test.sh` and `score.sh` select a dispatcher by the
dataset name, so adding a dataset does not require growing one large shared
script.

## Tree

```text
evaluation/
├── __init__.py
├── asv19.py        ASVspoof2019 test/score dispatcher
├── asv19_test.py   ASVspoof2019 inference and prediction CSVs
├── asv19_score.py  ASVspoof2019 metric integration
├── asv21.py        ASVspoof2021 test/score dispatcher
├── asv21_test.py   ASVspoof2021 LA/DF inference and CSVs
└── asv21_score.py  ASVspoof2021 official eval-package integration
```

## Dispatch flow

```text
test.sh DATASET [options]
        |
        v
evaluation/DATASET.py test
        |
        v
dataset-specific test module -> <run>/tested_results/*.csv

score.sh DATASET [options]
        |
        v
evaluation/DATASET.py score
        |
        v
dataset-specific score module -> tested_results/scored_results/
```

The test modules load a saved `ADDModel`, discover the requested checkpoint
variant, run the selected dataset, and save prediction/metric artifacts. The
score modules read those artifacts and invoke the corresponding local or
official metric implementation. Official ASVspoof resources remain under the
repository-level `official_scores/` directory and are not model code.
