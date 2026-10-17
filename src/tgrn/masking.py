from __future__ import annotations

import torch
from torch import Tensor


def apply_random_missingness(
    features: dict[str, Tensor],
    valid: dict[str, Tensor],
    rates: dict[str, float],
    *,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Mask valid time steps independently, using zero vectors as in the paper."""
    corrupted: dict[str, Tensor] = {}
    missing: dict[str, Tensor] = {}
    for modality, x in features.items():
        rate = float(rates.get(modality, 0.0))
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"missing rate for {modality} must be in [0, 1]")
        draws = torch.rand(valid[modality].shape, device=x.device, generator=generator)
        missing[modality] = (draws < rate) & valid[modality].bool()
        corrupted[modality] = x.masked_fill(missing[modality].unsqueeze(-1), 0.0)
    return corrupted, missing
