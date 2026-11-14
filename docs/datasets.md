# Datasets and preprocessing

## Custom JSON Lines dataset

`CustomVLADataset` accepts a directory containing `manifest.jsonl` or the path
to the manifest itself. Each non-empty line describes one single-step sample.
Image paths are relative to the manifest's directory.

Single-camera record:

```json
{"image":"images/front_0001.png","instruction":"pick up the cube","action":[0.1,0.2,0.3],"episode_id":"episode_1"}
```

Multi-camera record:

```json
{"images":{"front":"images/front_0001.png","wrist":"images/wrist_0001.png"},"instruction":"pick up the cube","action":[0.1,0.2,0.3],"proprioception":[0.0,0.1],"episode_id":"episode_1"}
```

Every record requires a non-empty instruction, at least one image path, and an
action vector matching the provided `ActionSpec`. Optional fields are
`proprioception` and `episode_id`. The current adapter loads image files when
the sample is requested and converts them to RGB.

```python
from vlaforge.core import ActionSpec
from vlaforge.data import CustomVLADataset

dataset = CustomVLADataset(
    "./data/train",
    action_spec=ActionSpec(
        dimensions=3,
        units=("meter", "meter", "radian"),
    ),
)
```

## Samples and batches

Each dataset sample uses `images`, `instruction`, and `action` keys. The
collator creates a `VLABatch` with image tensors shaped `(B, C, H, W)`, token
tensors shaped `(B, L)`, and single-step actions shaped `(B, D)`. All samples in
a batch must contain the same camera names and compatible tensor shapes.

Pass a tokenizer compatible with the chosen language encoder to `VLACollator`.
Pretrained tokenizers are not loaded automatically. Image normalization
defaults to channel-wise mean/std of 0.5 and resize to 224 by 224; pass an
`ImageTransform` with model-specific values when needed.

## Normalization

Fit action normalization using the training split only:

```python
from vlaforge.data import ActionNormalizer

normalizer = ActionNormalizer.from_actions(training_actions)
```

Use `normalizer.normalize` as the collator's `action_transform` and pass the
same normalizer to `Trainer` so its statistics are saved with checkpoints.
Validation data must not be used to fit statistics.

For a new adapter, implement `BaseVLADataset`, return the sample keys above,
provide an `ActionSpec`, and cover loading and validation behavior with small
synthetic tests.
