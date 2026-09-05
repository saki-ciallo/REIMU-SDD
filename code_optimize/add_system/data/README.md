# Data

This directory converts cached Hugging Face audio examples into the fixed
batch contract expected by `ADDModel`. It does not own model parameters.

## Files

```text
data/
├── __init__.py       public data API
├── datasets.py       load DatasetDict splits from disk
├── preprocessing.py  optional CPU RawBoost Dataset transforms
└── collation.py      stack waveforms and labels into tensors
```

## Processing tree

```text
DataSettings + Dataset cache
             |
             v
datasets.load_splits(...)
             |
             +--> train_dataset
             |
             +--> eval_dataset
                         |
                         v
       preprocessing.apply_rawboost_transforms(...)
       (only when augmentation.algorithm != 0)
                         |
                         v
                  Dataset.with_transform
                         |
                         v
             AudioClassificationCollator
                         |
                         v
             {input_values: [B, T], labels: [B]}
                         |
                         v
                       ADDModel
```

## Responsibilities

- `datasets.py` uses `datasets.load_from_disk` and requires a `DatasetDict`.
  It selects the configured train and validation split and optionally limits
  each split for smoke runs. It does not decode or reshape examples itself.
- `preprocessing.py` adapts the structured augmentation settings to the
  bundled original CPU RawBoost implementation. It attaches a transform with
  `Dataset.with_transform`, so augmentation is applied when examples are read
  and stays outside the model and CUDA graph. Validation augmentation is
  controlled separately by `apply_to_validation`.
- `collation.py` converts every one-dimensional waveform to `float32`, checks
  that all items in a batch have the same length, stacks them as `[B, T]`, and
  creates `long` labels. Variable-length batches are rejected because the
  current training contract uses fixed-length audio.

The train runner calls the modules in this order:

```text
load_splits -> apply_rawboost_transforms -> DataLoader/Trainer collator
```

The test CLIs use the same collator contract. Dataset conversion and fixed
length audio creation happen before this package, in the dataset preparation
tools; this directory only loads the resulting cache and applies optional
runtime transforms.
