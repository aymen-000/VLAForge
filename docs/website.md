# Using the documentation

Use these guides to move from installation to a working VLAForge application:

1. Follow [Installation](installation.md) to prepare Python and PyTorch.
2. Read [Getting started](getting_started.md) for a complete synthetic
   workflow that does not require pretrained weights or an external dataset.
3. Use the [dataset guide](datasets.md) to prepare observations and actions in
   the format VLAForge expects.
4. Follow [Training](training.md) to configure a model and train it.
5. Use [Inference and evaluation](inference.md) to load a checkpoint, predict
   actions, and measure results.
6. Consult the [API reference](api.md) for public classes, method signatures,
   and tensor shapes.

VLAForge supplies reusable interfaces and components. Applications provide
their chosen vision and language encoders, tokenizer, and dataset paths, so
the guides explain where those pieces connect to the framework.
