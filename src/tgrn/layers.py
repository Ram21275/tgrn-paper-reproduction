from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.encoding.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds configured maximum")
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class FeatureEncoder(nn.Module):
    """Paper equations (1)-(2): temporal Conv1D plus positional encoding."""

    def __init__(self, input_dim: int, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.position = SinusoidalPositionEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.position(x))


class AttentionBlock(nn.Module):
    """Post-norm attention + FFN block used for SA, TCA, and EA."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        key_valid: Tensor | None = None,
        query_valid: Tensor | None = None,
    ) -> Tensor:
        if key_valid is None:
            key_padding_mask = None
        else:
            safe_key_valid = key_valid.bool().clone()
            empty = ~safe_key_valid.any(dim=1)
            if empty.any():
                safe_key_valid[empty, 0] = True
            key_padding_mask = ~safe_key_valid
        attended, _ = self.attention(
            query, key, value, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(query + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        if query_valid is not None:
            x = x * query_valid.unsqueeze(-1).to(x.dtype)
        return x


class ThreeWaySE(nn.Module):
    """Sequence adaptation of the paper's GAP + XGMP + YGMP 3SEnet.

    The paper defines 3SEnet for a CxHxW tensor but does not specify the exact
    reshape from an unaligned sequence. Here BxTxD is treated as a single-channel
    T-by-D map. A global channel gate and two coordinate-preserving max gates are
    multiplied with the input, matching equations (16)-(21).
    """

    def __init__(self, d_model: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(d_model // reduction, 8)
        self.global_gate = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, d_model), nn.Sigmoid()
        )
        self.time_gate = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, d_model), nn.Sigmoid()
        )

    def forward(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        if valid is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        weights = valid.unsqueeze(-1).to(x.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        global_descriptor = (x * weights).sum(dim=1) / denom
        global_gate = self.global_gate(global_descriptor).unsqueeze(1)

        masked = x.masked_fill(~valid.unsqueeze(-1), -torch.inf)
        time_descriptor = masked.amax(dim=-1, keepdim=True)
        time_descriptor = torch.where(valid.unsqueeze(-1), time_descriptor, torch.zeros_like(time_descriptor))
        time_gate = self.time_gate(time_descriptor)

        feature_descriptor = masked.amax(dim=1)
        feature_descriptor = torch.where(torch.isfinite(feature_descriptor), feature_descriptor, torch.zeros_like(feature_descriptor))
        feature_gate = self.feature_gate(feature_descriptor).unsqueeze(1)
        return x * global_gate * time_gate * feature_gate * weights


def masked_mean(x: Tensor, valid: Tensor | None) -> Tensor:
    if valid is None:
        return x.mean(dim=1)
    weights = valid.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class PMAG(nn.Module):
    """Progressive Modality-mixing Adaptation Gate, equations (24)-(31)."""

    def __init__(self, d_model: int, phi: float = 0.2) -> None:
        super().__init__()
        self.phi = phi
        self.gate_1 = nn.Linear(2 * d_model, d_model)
        self.gate_2 = nn.Linear(2 * d_model, d_model)
        self.shift_1 = nn.Linear(d_model, d_model, bias=False)
        self.shift_2 = nn.Linear(d_model, d_model, bias=False)
        self.shift_bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, companion_1: Tensor, primary: Tensor, companion_2: Tensor) -> Tensor:
        gate_1 = torch.relu(self.gate_1(torch.cat([companion_1, primary], dim=-1)))
        gate_2 = torch.relu(self.gate_2(torch.cat([companion_2, primary], dim=-1)))
        shift = self.shift_1(gate_1 * companion_1) + self.shift_2(gate_2 * companion_2)
        shift = shift + self.shift_bias
        primary_norm = primary.norm(dim=-1, keepdim=True)
        shift_norm = shift.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = torch.clamp(self.phi * primary_norm / shift_norm, max=1.0)
        return primary + delta * shift
