"""Language preprocessing wrappers."""

from collections.abc import Callable, Sequence
from typing import Any


class TokenizerTransform:
    """Apply an injected batch tokenizer without binding to a tokenizer library."""

    def __init__(
        self,
        tokenizer: Callable[..., dict[str, Any]],
        max_length: int = 77,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, instructions: Sequence[str]) -> dict[str, Any]:
        """Tokenize a batch and return padded ``input_ids`` and ``attention_mask``."""
        if not instructions or any(not text.strip() for text in instructions):
            raise ValueError("instructions must be a non-empty batch of non-empty strings")
        result = self.tokenizer(
            list(instructions),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "input_ids" not in result or "attention_mask" not in result:
            raise ValueError("tokenizer must return input_ids and attention_mask")
        return result
