from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _as_float_tensor(value: Any) -> Tensor:
    array = np.asarray(value, dtype=np.float32)
    array[~np.isfinite(array)] = 0.0
    return torch.from_numpy(array)


class MultimodalPickleDataset(Dataset[dict[str, Tensor]]):
    """Loader for the pickle structure documented by the TgRN authors.

    This implementation expects precomputed text embeddings shaped [N,T,D].
    Token triplets in `text_bert` require a separate BERT feature extraction
    stage and are intentionally rejected with an actionable error.
    """

    def __init__(self, path: str | Path, split: str) -> None:
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if split not in payload:
            raise KeyError(f"split {split!r} not found; available: {sorted(payload)}")
        data = payload[split]
        text = np.asarray(data.get("text"))
        if text.ndim != 3:
            raise ValueError("`text` must contain precomputed embeddings with shape [N,T,D]")
        self.text = _as_float_tensor(text)
        self.audio = _as_float_tensor(data["audio"])
        self.vision = _as_float_tensor(data["vision"])
        labels = data.get("regression_labels", data.get("labels"))
        if labels is None:
            raise KeyError("expected `regression_labels` (or `labels`) in dataset split")
        self.labels = _as_float_tensor(labels).reshape(-1)
        self.audio_lengths = data.get("audio_lengths")
        self.vision_lengths = data.get("vision_lengths")
        if not (len(self.text) == len(self.audio) == len(self.vision) == len(self.labels)):
            raise ValueError("modalities and labels have inconsistent sample counts")

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def _valid_mask(x: Tensor, length: Any | None) -> Tensor:
        if length is not None:
            return torch.arange(x.size(0)) < int(length)
        return x.abs().sum(dim=-1).ne(0)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        text, audio, vision = self.text[index], self.audio[index], self.vision[index]
        return {
            "text": text,
            "audio": audio,
            "vision": vision,
            "text_valid": self._valid_mask(text, None),
            "audio_valid": self._valid_mask(audio, None if self.audio_lengths is None else self.audio_lengths[index]),
            "vision_valid": self._valid_mask(vision, None if self.vision_lengths is None else self.vision_lengths[index]),
            "label": self.labels[index],
        }
