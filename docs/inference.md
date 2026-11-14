# Inference and evaluation

## Loading a VLAActionModel checkpoint

Recreate encoder modules and tokenizer matching those used during training.
`from_pretrained` loads the saved VLAForge config and full model weights into
those modules:

```python
from vlaforge.inference import InferencePipeline
from vlaforge.models import VLAActionModel

model = VLAActionModel.from_pretrained(
    "checkpoints/vla_action/checkpoint_best",
    vision_encoder=vision_encoder,
    language_encoder=language_encoder,
    tokenizer=tokenizer,
)
pipeline = InferencePipeline(model, device="auto")
action = pipeline.predict_action(observation)
```

`load_model_weights` can load a Safetensors file or a directory into any
already-constructed `BaseVLA` model. The inference module does not import the
Trainer or download external encoder weights.

`Action.values` is a tensor. `Action.representation` and `Action.horizon`
describe how to interpret it. When the model's `ActionSpec.normalized` is true,
`InferencePipeline` requires the training `ActionNormalizer` or the checkpoint's
`normalization.json` and denormalizes predicted values automatically.

## Offline evaluator

`Evaluator` computes MSE, MAE, and RMSE across all predicted action values in a
dataset. It requires target actions in each batch. Metrics use values emitted
by the collator by default. To report physical units, pass the training
normalizer and set `metric_space="physical"`:

```python
from vlaforge.evaluation import Evaluator

results = Evaluator(
    model=model,
    dataset=validation_dataset,
    metric_space="physical",
    action_normalizer=normalizer,
).evaluate()
```

These are offline prediction metrics; they do not measure task success on a
robot or in a simulator.
