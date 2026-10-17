from __future__ import annotations

import torch

from tgrn import TgRN, TgRNConfig
from tgrn.masking import apply_random_missingness


def main() -> None:
    config = TgRNConfig(text_dim=32, audio_dim=5, vision_dim=20, d_model=32, num_heads=4, ff_dim=64)
    model = TgRN(config)
    full = {
        "text": torch.randn(4, 12, 32),
        "audio": torch.randn(4, 18, 5),
        "vision": torch.randn(4, 15, 20),
    }
    valid = {name: torch.ones(value.shape[:2], dtype=torch.bool) for name, value in full.items()}
    corrupted, missing = apply_random_missingness(full, valid, {name: 0.2 for name in full})
    output = model(**corrupted, **{f"{name}_valid": valid[name] for name in full})
    loss, parts = model.loss(output, torch.randn(4), full, missing, valid)
    loss.backward()
    print("prediction shape:", tuple(output.prediction.shape))
    print("losses:", {name: round(value.item(), 4) for name, value in parts.items()})


if __name__ == "__main__":
    main()
