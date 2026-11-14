# VLAForge

VLAForge is a modular Python framework for building Vision-Language-Action
models. It provides shared interfaces and data structures so model, dataset,
training, and inference implementations can evolve independently.

The project is at an early development stage. The package currently provides
core model contracts and a basic data pipeline: JSON Lines custom datasets,
image and tokenizer transforms, action normalization, and batch collation.
Model architectures and end-to-end training workflows will be introduced
incrementally. No pretrained model support is claimed yet.

## Custom dataset format

Pass either a directory containing `manifest.jsonl` or the manifest path to
`CustomVLADataset`. Each non-empty line is a JSON object with an instruction,
one or more image paths relative to the manifest, and an action vector:

```json
{"image": "images/frame_0001.png", "instruction": "pick up the cube", "action": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], "episode_id": "episode_1"}
```

For multiple cameras, use an `images` object keyed by camera name instead of
`image`. The dataset requires an `ActionSpec`; compute normalization statistics
from the training split only. Inject a tokenizer matching the chosen language
model into `VLACollator`.

## Development

The package uses a `src/` layout. Install it in editable mode with its
development dependencies using `pip install -e '.[dev]'`.

## License

The project license has not been selected yet.
