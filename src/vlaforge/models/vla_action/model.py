"""Backbone-agnostic VLM plus continuous action prediction model."""

from collections.abc import Callable, Mapping
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as functional
from torch import nn

from vlaforge.core import Action, ActionSpec, BaseVLA, Observation, VLABatch
from vlaforge.core.registry import register_model
from vlaforge.data.transforms import ImageTransform, TokenizerTransform
from vlaforge.models.vla_action.config import VLAActionConfig


def _extract_features(output: Any, keys: tuple[str, ...]) -> torch.Tensor:
    """Extract a tensor feature representation from common encoder outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping):
        for key in keys:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        raise TypeError(f"encoder mapping must contain one of {keys}")
    for key in keys:
        value = getattr(output, key, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("encoder output must expose a tensor feature representation")


def _pool_features(
    features: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert pooled, token, or spatial encoder outputs to ``(B, D)``."""
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        if attention_mask is None:
            return features.mean(dim=1)
        mask = attention_mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
        return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if features.ndim == 4:
        return features.mean(dim=(-2, -1))
    raise ValueError("encoder features must have shape (B,D), (B,L,D), or (B,C,H,W)")


@register_model("vla_action")
class VLAActionModel(BaseVLA):
    """Fuse image and language features, then predict continuous actions.

    Encoders are injected as ``torch.nn.Module`` instances. Their outputs may
    be pooled feature tensors, token tensors, or common model output objects
    exposing ``pooler_output`` or ``last_hidden_state``. Language encoders must
    accept ``input_ids`` and ``attention_mask`` keyword arguments.

    Args:
        vision_encoder: Module accepting a camera image batch.
        language_encoder: Module accepting token IDs and attention masks.
        config: Architecture dimensions and action semantics.
        tokenizer: Optional batch tokenizer used by :meth:`predict_action`.
        image_transform: Optional transform used by :meth:`predict_action`.
        text_transform: Optional transform overriding ``tokenizer``.
    """

    model_type = "vla_action"

    def __init__(
        self,
        vision_encoder: nn.Module,
        language_encoder: nn.Module,
        config: VLAActionConfig,
        tokenizer: Callable[..., dict[str, Any]] | None = None,
        image_transform: Callable[[Any], torch.Tensor] | None = None,
        text_transform: Callable[[list[str]], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.config = config
        self.image_transform = image_transform or ImageTransform()
        self.text_transform = text_transform or (
            TokenizerTransform(tokenizer) if tokenizer is not None else None
        )
        input_dim = config.vision_feature_dim + config.language_feature_dim
        self.action_head = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.action_spec.dimensions),
        )
        self._encoders_trainable = not config.freeze_encoders
        self.set_encoder_trainability(not config.freeze_encoders)

    def set_encoder_trainability(self, trainable: bool) -> None:
        """Enable or freeze gradients for both pretrained encoders."""
        self._encoders_trainable = trainable
        for encoder in (self.vision_encoder, self.language_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(trainable)
        if not trainable:
            self.vision_encoder.eval()
            self.language_encoder.eval()

    def train(self, mode: bool = True) -> "VLAActionModel":
        """Set training mode while keeping frozen encoders in evaluation mode."""
        super().train(mode)
        if not self._encoders_trainable:
            self.vision_encoder.eval()
            self.language_encoder.eval()
        return self

    def _encode(self, batch: VLABatch) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.camera_name in batch.images:
            images = batch.images[self.config.camera_name]
        elif len(batch.images) == 1:
            images = next(iter(batch.images.values()))
        else:
            raise ValueError(
                f"camera {self.config.camera_name!r} is missing from batch; "
                f"available cameras: {', '.join(batch.images)}"
            )

        vision_output = self.vision_encoder(images)
        language_output = self.language_encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        vision = _pool_features(
            _extract_features(
                vision_output,
                ("image_embeds", "pooler_output", "last_hidden_state", "features"),
            )
        )
        language = _pool_features(
            _extract_features(
                language_output,
                ("text_embeds", "pooler_output", "last_hidden_state", "features"),
            ),
            attention_mask=batch.attention_mask,
        )
        if vision.shape[-1] != self.config.vision_feature_dim:
            raise ValueError(
                "vision encoder feature dimension mismatch: expected "
                f"{self.config.vision_feature_dim}, got {vision.shape[-1]}"
            )
        if language.shape[-1] != self.config.language_feature_dim:
            raise ValueError(
                "language encoder feature dimension mismatch: expected "
                f"{self.config.language_feature_dim}, got {language.shape[-1]}"
            )
        if vision.shape[0] != language.shape[0]:
            raise ValueError("vision and language encoders returned different batch sizes")
        return vision, language

    def forward(self, batch: VLABatch) -> torch.Tensor:
        """Predict a batch of actions with shape ``(B, action_dimensions)``."""
        vision, language = self._encode(batch)
        return self.action_head(torch.cat((vision, language), dim=-1))

    def compute_loss(self, batch: VLABatch) -> torch.Tensor:
        """Compute mean squared error for continuous target actions."""
        if batch.actions is None:
            raise ValueError("batch.actions is required to compute supervised loss")
        predictions = self(batch)
        targets = batch.actions.to(device=predictions.device, dtype=predictions.dtype)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"prediction shape {tuple(predictions.shape)} does not match "
                f"target shape {tuple(targets.shape)}"
            )
        return functional.mse_loss(predictions, targets)

    def predict_action(self, observation: Observation) -> Action:
        """Predict one action from an unbatched observation without gradients."""
        if self.text_transform is None:
            raise RuntimeError(
                "predict_action requires a tokenizer or text_transform at model construction"
            )
        if self.config.camera_name in observation.images:
            image = observation.images[self.config.camera_name]
        elif len(observation.images) == 1:
            image = next(iter(observation.images.values()))
        else:
            raise ValueError(
                f"camera {self.config.camera_name!r} is missing from observation"
            )

        device = next(self.parameters()).device
        image_tensor = self.image_transform(image)
        if image_tensor.ndim != 3:
            raise ValueError("image_transform must return a tensor shaped (C,H,W)")
        tokens = self.text_transform([observation.instruction])
        try:
            input_ids = torch.as_tensor(tokens["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.as_tensor(
                tokens["attention_mask"], dtype=torch.long, device=device
            )
        except KeyError as error:
            raise ValueError("text_transform must return input_ids and attention_mask") from error

        batch = VLABatch(
            images={self.config.camera_name: image_tensor.unsqueeze(0).to(device)},
            input_ids=input_ids,
            attention_mask=attention_mask,
            proprioception=(
                observation.proprioception.unsqueeze(0).to(device)
                if observation.proprioception is not None
                else None
            ),
        )
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                values = self(batch).squeeze(0).detach().cpu()
        finally:
            self.train(was_training)
        return Action(
            values=values,
            representation=self.config.action_spec.representation,
            horizon=1,
        )

    def get_action_spec(self) -> ActionSpec:
        """Return the configured output action specification."""
        return self.config.action_spec

    def save_pretrained(self, directory: str | Path) -> None:
        """Save weights and architecture configuration for later reconstruction.

        Encoder implementations are supplied by the caller when loading; the
        state dictionary includes their weights along with the action head.
        """
        path = Path(directory).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        state = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in self.state_dict().items()
        }
        save_file(state, str(path / "model.safetensors"))
        (path / "config.json").write_text(
            json.dumps(
                {"model_type": self.model_type, "config": asdict(self.config)},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        vision_encoder: nn.Module,
        language_encoder: nn.Module,
        tokenizer: Callable[..., dict[str, Any]] | None = None,
        image_transform: Callable[[Any], torch.Tensor] | None = None,
        text_transform: Callable[[list[str]], Mapping[str, Any]] | None = None,
        device: torch.device | str = "cpu",
    ) -> "VLAActionModel":
        """Rebuild a model from saved settings and weights.

        The encoder modules must be constructed by the application. The
        checkpoint supplies their trained state and the VLAForge action-head
        configuration.
        """
        path = Path(directory).expanduser()
        config_path = path / "config.json"
        weights_path = path / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"model directory must contain config.json and model.safetensors: {path}"
            )
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved.get("model_type") != cls.model_type:
            raise ValueError(f"checkpoint model_type must be {cls.model_type!r}")
        config_data = saved.get("config")
        if not isinstance(config_data, dict):
            raise ValueError("checkpoint config.json must contain a config object")
        action_data = config_data.get("action_spec")
        if not isinstance(action_data, dict):
            raise ValueError("checkpoint config must include action_spec")
        action_data["units"] = (
            tuple(action_data["units"]) if action_data.get("units") is not None else None
        )
        config_data["action_spec"] = ActionSpec(**action_data)
        model = cls(
            vision_encoder=vision_encoder,
            language_encoder=language_encoder,
            config=VLAActionConfig(**config_data),
            tokenizer=tokenizer,
            image_transform=image_transform,
            text_transform=text_transform,
        )
        model.load_state_dict(load_file(str(weights_path), device=str(device)))
        return model.to(device)
