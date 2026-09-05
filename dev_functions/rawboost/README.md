# RawBoost Experiments

- `cuda/` contains the experimental GPU implementation and comparison script.
- `optimize/` contains the separate optimized RawBoost package and its tests.

Neither implementation is part of the default training path. Normal training
uses the original implementation in `utilis/RawBoost.py` when RawBoost is
enabled in the training configuration.
