# Training

This directory adapts the model and data components to the Hugging Face
Trainer lifecycle. It owns optimization-time behavior, while `models/` stays
loss-agnostic.

## Files

```text
training/
├── __init__.py   public loss and Trainer API
├── arguments.py  ExperimentSettings -> TrainingArguments
├── losses.py     weighted cross-entropy and focal loss
├── trainer.py    ADDTrainer and evaluation metrics
└── run.py        end-to-end training assembly and lifecycle
```

## Training assembly

```text
ResolvedExperiment
       |
       v
build_experiment_settings
       |
       +--> build_training_arguments
       +--> ADDModel
       +--> load_splits
       +--> apply_rawboost_transforms
       +--> AudioClassificationCollator
       +--> build_loss
       |
       v
ADDTrainer
       |
       v
Trainer.train / evaluate / checkpoint save
```

`run.py` also writes the resolved configuration, model report, run manifest,
and optional early-stopping callback. CUDA availability is checked before
model construction because the optimized FLA path has no CPU fallback.

## Loss and Trainer boundary

```text
ADDModel.forward
    -> logits / optional loss_logits / optional aux_loss
                              |
                              v
ADDTrainer.compute_loss
    -> selected loss(logits, labels)
    -> + MoE auxiliary loss when returned
```

`losses.py` selects weighted `CrossEntropyLoss` or numerically stable focal
loss from `LossSettings`. `trainer.py` passes `labels` to the model only so
AMSoftmax can create margin-adjusted `loss_logits`; it does not let the model
own the final loss calculation. Prediction metrics use the ordinary classifier
logits.

`arguments.py` maps optimizer, learning rate, weight decay, Adam betas,
cosine/warmup schedule, BF16, compilation, evaluation, checkpoint, and
dataloader settings to `TrainingArguments`. The YAML settings are therefore
the source of truth for a run rather than ad-hoc Trainer construction in
dataset-specific scripts.
