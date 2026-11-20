# Public API reference

This page summarizes supported Python interfaces. All package imports use the
`vlaforge` namespace.

## Core: `vlaforge.core`

- `Observation(images, instruction, proprioception=None, metadata={})`: one
  input observation. `images` maps camera names to image objects or tensors.
- `Action(values, representation="continuous", horizon=1)`: predicted action
  values and their interpretation.
- `ActionSpec(dimensions, representation="continuous", units=None,
  coordinate_frame=None, normalized=False)`: action dimensions and semantics.
- `VLABatch(images, input_ids, attention_mask, actions=None,
  proprioception=None, metadata=None)`: collated model input. Per-camera images
  are `(B,C,H,W)`, tokens are `(B,L)`, and single-step actions are `(B,D)`.
- `BaseVLA`: abstract `torch.nn.Module` contract requiring `forward(batch)`,
  `predict_action(observation)`, `compute_loss(batch)`, and
  `get_action_spec()`.
- `ModelRegistry` and `register_model(name)`: register and retrieve model
  implementations. Names must be unique.

## Data: `vlaforge.data`

- `BaseVLADataset`: dataset interface with `get_statistics()` and
  `get_action_spec()` in addition to PyTorch dataset methods.
- `CustomVLADataset(root, action_spec)`: JSON Lines manifest adapter.
- `LeRobotDatasetAdapter(action_spec, repo_id=None, field_mapping=None, ...)`:
  optional adapter for LeRobot 0.4.x datasets. Install the `lerobot` extra to
  construct from a repository ID. `LeRobotFieldMapping` maps camera, task,
  action, proprioception, and episode fields. `split_by_episode()` creates
  episode-disjoint train and validation subsets.
- `VLACollator(tokenizer=None, image_transform=None, text_transform=None,
  action_transform=None)`: creates `VLABatch` objects.
- `ImageTransform(size=(224,224), mean=(0.5,)*3, std=(0.5,)*3)`: converts an
  image to normalized RGB `(3,H,W)` tensor.
- `TokenizerTransform(tokenizer, max_length=77)`: calls an injected batched
  tokenizer and returns `input_ids` and `attention_mask`.
- `ActionNormalizer(mean, std)`: normalize/denormalize actions or fit with
  `ActionNormalizer.from_actions(training_actions)`.

## Model: `vlaforge.models`

- `VLAActionConfig(vision_feature_dim, language_feature_dim, action_spec,
  hidden_dim=512, dropout=0.1, camera_name="front", freeze_encoders=True)`.
- `VLAActionModel(vision_encoder, language_encoder, config, tokenizer=None,
  image_transform=None, text_transform=None)`: fuses encoder features and
  predicts continuous actions.
- `build_model(config, **components)`: build a registered model from a mapping
  or YAML path. The `vla_action` config uses `vision_feature_dim`,
  `language_feature_dim`, and an `action_spec` mapping. Supply
  `vision_encoder` and `language_encoder` as keyword components; optional
  tokenizer and transforms can also be supplied. VLAForge does not download
  or choose pretrained encoders automatically.
- `save_pretrained(directory)` and `VLAActionModel.from_pretrained(directory,
  vision_encoder=..., language_encoder=..., tokenizer=...)`: save and restore
  VLAForge weights/config. Encoder modules are constructed by the application.

Vision encoders may return `(B,D)`, `(B,L,D)`, or `(B,C,H,W)` tensors, or
objects exposing `image_embeds`, `pooler_output`, `last_hidden_state`, or
`features`. Language encoders accept `input_ids` and `attention_mask`, and may
return corresponding text/pooler/hidden-state features. Configured feature
dimensions must match the pooled encoder outputs.

## Training: `vlaforge.training`

- `TrainingConfig`: validated settings; construct from a mapping or YAML path
  with `TrainingConfig.from_file(path)`.
- `Trainer(model, train_dataset, val_dataset=None, config=None, collator=None,
  resume_from=None, action_normalizer=None)`: trains a `BaseVLA` through its
  `compute_loss` method. `train()` returns per-epoch metric dictionaries.

## Inference: `vlaforge.inference`

- `load_model_weights(model, checkpoint, device="auto", strict=True)`: restore
  a Safetensors state dict into an existing model.
- `InferencePipeline(model, checkpoint=None, device="auto",
  action_normalizer=None)`: predict `Action` values from `Observation` inputs.

## Evaluation: `vlaforge.evaluation`

- `Evaluator(model, dataset, collator=None, batch_size=8, num_workers=0,
  device="auto", metric_space="normalized", action_normalizer=None)`.
- `evaluate()` returns `{"mse": float, "mae": float, "rmse": float}`.
- `mean_squared_error`, `mean_absolute_error`, and
  `root_mean_squared_error` are also available as standalone tensor metrics.
