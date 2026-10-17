from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TgRNConfig:
    """Model hyperparameters.

    Input dimensions follow the paper's extracted feature dimensions. The
    remaining defaults are conservative reproduction choices because the paper
    does not publish a complete architecture table.
    """

    text_dim: int = 768
    audio_dim: int = 5
    vision_dim: int = 20
    d_model: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    max_seq_len: int = 512
    se_reduction: int = 4
    pmag_phi: float = 0.2
    classifier_hidden: int = 128
    reconstruction_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 < self.pmag_phi <= 1.0:
            raise ValueError("pmag_phi must be in (0, 1]")
        if len(self.reconstruction_weights) != 3:
            raise ValueError("reconstruction_weights must contain text/audio/vision weights")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TgRNConfig":
        values = dict(values)
        if "reconstruction_weights" in values:
            values["reconstruction_weights"] = tuple(values["reconstruction_weights"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
