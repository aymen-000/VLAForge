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

## LeRobot datasets

Install the optional integration with `pip install -e '.[lerobot]'`. The base
VLAForge install does not import or require LeRobot. The optional extra pins
LeRobot 0.4.x, which supports Python 3.10. The adapter opens datasets lazily:

```python
from vlaforge.core import ActionSpec
from vlaforge.data import LeRobotDatasetAdapter, LeRobotFieldMapping

dataset = LeRobotDatasetAdapter(
    repo_id="lerobot/your_dataset",
    action_spec=ActionSpec(dimensions=7),
    field_mapping=LeRobotFieldMapping(
        camera_fields={"front": "observation.images.front"},
        task_field="task_index",
        action_field="action",
        proprioception_field="observation.state",
    ),
)
sample = dataset[0]
```

If camera keys are present in LeRobot metadata, `camera_fields` can be omitted
and the adapter infers them. The task, action, proprioception, and episode
field names can be changed to match a dataset. Supply an `ActionSpec` yourself
because feature dimensions do not identify physical units or coordinate frame.

`get_statistics()` computes action statistics over the adapter's selected
frames. Pass frame indices to compute statistics for a training subset only.
It reads an action-only column view when the LeRobot backend supports it;
otherwise it scans samples and may decode images.
`split_by_episode(validation_fraction=0.2, seed=42)` returns train and
validation `Subset`s with complete episodes kept together; it requires episode
range metadata. To fit normalization only on the training episodes:

```python
from vlaforge.data import ActionNormalizer

train_data, validation_data = dataset.split_by_episode(seed=42)
statistics = dataset.get_statistics(indices=train_data.indices)
normalizer = ActionNormalizer(
    statistics["action_mean"],
    statistics["action_std"],
)
```
