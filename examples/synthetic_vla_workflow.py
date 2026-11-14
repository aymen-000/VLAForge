"""Run a small synthetic train, checkpoint, inference, and evaluation workflow."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from vlaforge import ActionSpec
from vlaforge.core import Observation
from vlaforge.evaluation import Evaluator
from vlaforge.inference import InferencePipeline
from vlaforge.models import VLAActionConfig, VLAActionModel
from vlaforge.training import Trainer, TrainingConfig


class SyntheticRobotDataset(Dataset[dict[str, object]]):
    """Small in-memory dataset that requires no downloads or robot data."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "images": {
                "front": Image.new("RGB", (32, 32), color=(index * 20 % 255, 64, 128))
            },
            "instruction": f"move object {index % 3}",
            "action": torch.tensor([index / max(self.size, 1), (index % 3) / 3]),
        }


class MeanVisionEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(-2, -1))


class TinyLanguageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def tokenizer(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
    del kwargs
    token_ids = torch.tensor(
        [[len(text) % 7 + 1, 2, 3, 0] for text in texts], dtype=torch.long
    )
    return {
        "input_ids": token_ids,
        "attention_mask": (token_ids != 0).long(),
    }


def build_model() -> VLAActionModel:
    return VLAActionModel(
        vision_encoder=MeanVisionEncoder(),
        language_encoder=TinyLanguageEncoder(),
        config=VLAActionConfig(
            vision_feature_dim=3,
            language_feature_dim=4,
            action_spec=ActionSpec(dimensions=2),
            hidden_dim=16,
            freeze_encoders=True,
        ),
        tokenizer=tokenizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./outputs/synthetic_vla")
    )
    args = parser.parse_args()

    train_dataset = SyntheticRobotDataset(size=12)
    validation_dataset = SyntheticRobotDataset(size=4)
    model = build_model()
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        config=TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            scheduler="none",
            output_dir=str(args.output_dir),
        ),
    )
    history = trainer.train()
    checkpoint = args.output_dir / "checkpoint_last"

    restored_model = VLAActionModel.from_pretrained(
        checkpoint,
        vision_encoder=MeanVisionEncoder(),
        language_encoder=TinyLanguageEncoder(),
        tokenizer=tokenizer,
    )
    pipeline = InferencePipeline(restored_model, device="cpu")
    action = pipeline.predict_action(
        Observation(
            images={"front": Image.new("RGB", (32, 32), color=(64, 64, 128))},
            instruction="move object 1",
        )
    )
    results = Evaluator(
        model=restored_model,
        dataset=validation_dataset,
        device="cpu",
    ).evaluate()
    print({"epochs": len(history), "predicted_action": action.values.tolist()})
    print(results)


if __name__ == "__main__":
    main()
