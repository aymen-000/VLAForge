"""Standalone checkpoint loading and action prediction."""

from pathlib import Path

from safetensors.torch import load_file
import torch

from vlaforge.core import Action, BaseVLA, Observation
from vlaforge.data import ActionNormalizer


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {device}")
    return resolved


def load_model_weights(
    model: BaseVLA,
    checkpoint: str | Path,
    device: str | torch.device = "auto",
    strict: bool = True,
) -> BaseVLA:
    """Load Safetensors weights into an already constructed model.

    ``checkpoint`` may be a model file or a directory containing
    ``model.safetensors``. Construction is deliberately left to the caller so
    external encoders and their optional dependencies stay application-owned.
    """
    resolved_device = resolve_device(device)
    path = Path(checkpoint).expanduser()
    weights_path = path / "model.safetensors" if path.is_dir() else path
    if not weights_path.is_file():
        raise FileNotFoundError(f"model weights not found: {weights_path}")
    model.to(resolved_device)
    state = load_file(str(weights_path), device=str(resolved_device))
    model.load_state_dict(state, strict=strict)
    return model


class InferencePipeline:
    """Run action prediction without importing the training package."""

    def __init__(
        self,
        model: BaseVLA,
        checkpoint: str | Path | None = None,
        device: str | torch.device = "auto",
        action_normalizer: ActionNormalizer | None = None,
    ) -> None:
        self.device = resolve_device(device)
        if checkpoint is not None:
            load_model_weights(model, checkpoint, device=self.device)
            if action_normalizer is None:
                checkpoint_path = Path(checkpoint).expanduser()
                checkpoint_dir = (
                    checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
                )
                normalization_path = checkpoint_dir / "normalization.json"
                if normalization_path.is_file():
                    action_normalizer = ActionNormalizer.load(normalization_path)
        else:
            model.to(self.device)
        self.model = model.eval()
        self.action_normalizer = action_normalizer

    @torch.inference_mode()
    def predict_action(self, observation: Observation) -> Action:
        """Return the model's standardized action for one observation."""
        action = self.model.predict_action(observation)
        if not isinstance(action, Action):
            raise TypeError("BaseVLA.predict_action must return an Action")
        if self.model.get_action_spec().normalized:
            if self.action_normalizer is None:
                raise ValueError(
                    "model actions are normalized; provide the training ActionNormalizer"
                )
            action.values = self.action_normalizer.denormalize(action.values)
        return action
