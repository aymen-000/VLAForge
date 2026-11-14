# Getting started

The executable synthetic workflow is the shortest complete example:

```bash
python examples/synthetic_vla_workflow.py
```

It creates a small in-memory dataset, injects tiny encoders and a tokenizer,
trains the VLA action model, reloads a checkpoint, predicts one action, and
evaluates a held-out dataset. It does not download data or pretrained weights.

## Main components

1. `CustomVLADataset` or another `BaseVLADataset` returns images, instructions,
   and target actions.
2. `VLACollator` applies image/text transforms and builds `VLABatch` instances.
3. A `BaseVLA` implementation predicts actions and owns its loss function.
4. `Trainer` optimizes any `BaseVLA` model through `compute_loss`.
5. A model checkpoint can be loaded for inference and offline evaluation.

Read [Datasets](datasets.md), [Training](training.md), [Inference](inference.md),
and the [API guide](api.md) for details.
