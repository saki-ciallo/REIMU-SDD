from __future__ import annotations

from transformers import TrainingArguments

from ..configuration.settings import ExperimentSettings


def build_training_arguments(settings: ExperimentSettings) -> TrainingArguments:
    training = settings.training
    return TrainingArguments(
        output_dir=str(settings.output_dir),
        do_train=training.do_train,
        do_eval=training.do_eval,
        seed=settings.run.seed,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        num_train_epochs=training.num_train_epochs,
        max_steps=training.max_steps,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        optim=training.optimizer,
        adam_beta1=training.adam_beta1,
        adam_beta2=training.adam_beta2,
        lr_scheduler_type=training.lr_scheduler_type,
        # Transformers 5.14 accepts a float ratio through warmup_steps.
        warmup_steps=training.warmup_ratio,
        bf16=training.bf16,
        bf16_full_eval=training.bf16,
        torch_compile=training.torch_compile,
        torch_compile_backend="inductor" if training.torch_compile else None,
        torch_compile_mode=(training.torch_compile_mode if training.torch_compile else None),
        eval_strategy=training.eval_strategy if training.do_eval else "no",
        save_strategy=training.save_strategy,
        logging_strategy=training.logging_strategy,
        logging_steps=training.logging_steps,
        save_total_limit=training.save_total_limit,
        load_best_model_at_end=training.load_best_model_at_end,
        metric_for_best_model=training.metric_for_best_model,
        greater_is_better=training.greater_is_better,
        dataloader_num_workers=training.dataloader_num_workers,
        dataloader_persistent_workers=training.dataloader_num_workers > 0,
        remove_unused_columns=training.remove_unused_columns,
        label_names=["labels"],
        report_to=training.report_to or [],
    )
