# Training and checkpoints

`Trainer` is architecture-independent. It calls the model's `compute_loss`,
uses AdamW, supports cosine or step scheduling, reports validation loss, and
saves Safetensors model weights plus optimizer/scheduler state for resumption.

```python
from vlaforge.training import Trainer

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=validation_dataset,
    config="configs/training/default.yaml",
)
history = trainer.train()
```

The model must be constructed before the Trainer. `VLAActionModel` can provide
its own tokenizer and transforms, which the Trainer uses to infer a collator.
For another model, pass `collator=` explicitly. The collator must return a
`VLABatch` with target actions for supervised training.

## Configuration

`TrainingConfig` accepts a dataclass instance, a mapping, or a YAML path. The
provided `configs/training/default.yaml` demonstrates `epochs`, `batch_size`,
`learning_rate`, `weight_decay`, `optimizer`, `scheduler`, validation interval,
device, seed, checkpoint directory, and best-checkpoint behavior. Only AdamW
is currently supported. Scheduler choices are `none`, `cosine`, and `step`.

`device: auto` chooses CUDA if available and CPU otherwise. `num_workers`
defaults to zero for portability. Gradient clipping is optional through
`max_grad_norm`.

## Checkpoint contents

The output directory contains `checkpoint_last/` and, when validation improves
and `save_best` is enabled, `checkpoint_best/`. A VLAActionModel checkpoint
contains:

- `model.safetensors`: model parameters.
- `config.json`: model type and architecture configuration.
- `training_state.pt`: optimizer, scheduler, epoch, step, and best validation
  loss for resumption.
- `metadata.json`: model/action details and training configuration.
- `normalization.json`: present when an `ActionNormalizer` is passed to
  `Trainer`.

Resume with the same model architecture and encoder modules:

```python
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=validation_dataset,
    config="configs/training/default.yaml",
    resume_from="checkpoints/vla_action/checkpoint_last",
)
trainer.train()
```

Checkpoints do not instantiate external encoders for you. Recreate the model
with compatible encoder modules before resuming.
