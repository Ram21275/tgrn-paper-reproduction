from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import TgRNConfig
from .layers import AttentionBlock, FeatureEncoder, PMAG, ThreeWaySE, masked_mean


@dataclass
class TgRNOutput:
    prediction: Tensor
    reconstructions: dict[str, Tensor]
    embeddings: dict[str, Tensor]


class TextGuidedExtractionModule(nn.Module):
    def __init__(self, config: TgRNConfig) -> None:
        super().__init__()
        c = config
        self.encoders = nn.ModuleDict({
            "text": FeatureEncoder(c.text_dim, c.d_model, c.dropout, c.max_seq_len),
            "audio": FeatureEncoder(c.audio_dim, c.d_model, c.dropout, c.max_seq_len),
            "vision": FeatureEncoder(c.vision_dim, c.d_model, c.dropout, c.max_seq_len),
        })
        args = (c.d_model, c.num_heads, c.ff_dim, c.dropout)
        self.self_attention = nn.ModuleDict({m: AttentionBlock(*args) for m in ("text", "audio", "vision")})
        self.from_text = nn.ModuleDict({m: AttentionBlock(*args) for m in ("audio", "vision")})
        self.to_text = nn.ModuleDict({m: AttentionBlock(*args) for m in ("audio", "vision")})
        self.mix = nn.ModuleDict({
            "text": nn.Linear(3 * c.d_model, c.d_model),
            "audio": nn.Linear(2 * c.d_model, c.d_model),
            "vision": nn.Linear(2 * c.d_model, c.d_model),
        })

    def forward(self, features: dict[str, Tensor], valid: dict[str, Tensor]) -> dict[str, Tensor]:
        encoded = {m: self.encoders[m](features[m]) for m in features}
        intra = {
            m: self.self_attention[m](x, x, x, key_valid=valid[m], query_valid=valid[m])
            for m, x in encoded.items()
        }
        text = intra["text"]
        guided: dict[str, Tensor] = {}
        text_enriched: dict[str, Tensor] = {}
        for modality in ("audio", "vision"):
            guided[modality] = self.from_text[modality](
                intra[modality], text, text, key_valid=valid["text"], query_valid=valid[modality]
            )
            text_enriched[modality] = self.to_text[modality](
                text, intra[modality], intra[modality],
                key_valid=valid[modality], query_valid=valid["text"]
            )
        return {
            "text": self.mix["text"](torch.cat([text, text_enriched["audio"], text_enriched["vision"]], dim=-1)),
            "audio": self.mix["audio"](torch.cat([intra["audio"], guided["audio"]], dim=-1)),
            "vision": self.mix["vision"](torch.cat([intra["vision"], guided["vision"]], dim=-1)),
        }


class ReconstructionModule(nn.Module):
    def __init__(self, config: TgRNConfig) -> None:
        super().__init__()
        c = config
        args = (c.d_model, c.num_heads, c.ff_dim, c.dropout)
        self.enhance = nn.ModuleDict({m: AttentionBlock(*args) for m in ("text", "audio", "vision")})
        self.three_se = nn.ModuleDict({m: ThreeWaySE(c.d_model, c.se_reduction) for m in ("text", "audio", "vision")})
        self.output = nn.ModuleDict({
            "text": nn.Linear(c.d_model, c.text_dim),
            "audio": nn.Linear(c.d_model, c.audio_dim),
            "vision": nn.Linear(c.d_model, c.vision_dim),
        })

    def forward(self, embeddings: dict[str, Tensor], valid: dict[str, Tensor]) -> dict[str, Tensor]:
        reconstructed = {}
        for modality, x in embeddings.items():
            enhanced = self.enhance[modality](x, x, x, key_valid=valid[modality], query_valid=valid[modality])
            reconstructed[modality] = self.output[modality](self.three_se[modality](enhanced, valid[modality]))
        return reconstructed


class TextGuidedFusionModule(nn.Module):
    def __init__(self, config: TgRNConfig) -> None:
        super().__init__()
        c = config
        self.audio_pmag = PMAG(c.d_model, c.pmag_phi)
        self.vision_pmag = PMAG(c.d_model, c.pmag_phi)
        self.text_pmag = PMAG(c.d_model, c.pmag_phi)
        self.classifier = nn.Sequential(
            nn.Linear(c.d_model, c.classifier_hidden),
            nn.LeakyReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.classifier_hidden, 1),
        )

    def forward(self, embeddings: dict[str, Tensor], valid: dict[str, Tensor]) -> Tensor:
        pooled = {m: masked_mean(x, valid[m]) for m, x in embeddings.items()}
        shifted_audio = self.audio_pmag(pooled["vision"], pooled["audio"], pooled["text"])
        shifted_vision = self.vision_pmag(pooled["audio"], pooled["vision"], pooled["text"])
        shifted_text = self.text_pmag(shifted_audio, pooled["text"], shifted_vision)
        return self.classifier(shifted_text).squeeze(-1)


class TgRN(nn.Module):
    def __init__(self, config: TgRNConfig) -> None:
        super().__init__()
        self.config = config
        self.tem = TextGuidedExtractionModule(config)
        self.rm = ReconstructionModule(config)
        self.tfm = TextGuidedFusionModule(config)

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        *,
        text_valid: Tensor | None = None,
        audio_valid: Tensor | None = None,
        vision_valid: Tensor | None = None,
    ) -> TgRNOutput:
        features = {"text": text, "audio": audio, "vision": vision}
        supplied = {"text": text_valid, "audio": audio_valid, "vision": vision_valid}
        valid = {
            m: supplied[m].bool() if supplied[m] is not None else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            for m, x in features.items()
        }
        embeddings = self.tem(features, valid)
        reconstructions = self.rm(embeddings, valid)
        prediction = self.tfm(embeddings, valid)
        return TgRNOutput(prediction, reconstructions, embeddings)

    def loss(
        self,
        output: TgRNOutput,
        target: Tensor,
        originals: dict[str, Tensor],
        missing: dict[str, Tensor],
        valid: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        task = F.l1_loss(output.prediction, target.reshape_as(output.prediction))
        reconstruction: dict[str, Tensor] = {}
        for modality in ("text", "audio", "vision"):
            positions = missing[modality].bool() & valid[modality].bool()
            if positions.any():
                reconstruction[modality] = F.smooth_l1_loss(
                    output.reconstructions[modality][positions], originals[modality][positions]
                )
            else:
                reconstruction[modality] = task.new_zeros(())
        weights = self.config.reconstruction_weights
        total = task + sum(weights[i] * reconstruction[m] for i, m in enumerate(("text", "audio", "vision")))
        parts = {"total": total.detach(), "task": task.detach()}
        parts.update({f"reconstruction_{m}": v.detach() for m, v in reconstruction.items()})
        return total, parts
