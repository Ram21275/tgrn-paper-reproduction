import torch

from tgrn import TgRN, TgRNConfig
from tgrn.masking import apply_random_missingness


def test_forward_and_backward() -> None:
    config = TgRNConfig(text_dim=12, audio_dim=5, vision_dim=7, d_model=16, num_heads=4, ff_dim=32, classifier_hidden=8)
    model = TgRN(config)
    features = {
        "text": torch.randn(3, 6, 12),
        "audio": torch.randn(3, 8, 5),
        "vision": torch.randn(3, 7, 7),
    }
    valid = {m: torch.ones(x.shape[:2], dtype=torch.bool) for m, x in features.items()}
    corrupted, missing = apply_random_missingness(features, valid, {m: 0.3 for m in features}, generator=torch.Generator().manual_seed(1))
    output = model(**corrupted, **{f"{m}_valid": valid[m] for m in features})
    assert output.prediction.shape == (3,)
    assert {m: tuple(output.reconstructions[m].shape) for m in features} == {m: tuple(x.shape) for m, x in features.items()}
    loss, parts = model.loss(output, torch.randn(3), features, missing, valid)
    assert torch.isfinite(loss)
    assert "reconstruction_text" in parts
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_padding_masks_are_supported() -> None:
    config = TgRNConfig(text_dim=4, audio_dim=3, vision_dim=2, d_model=8, num_heads=2, ff_dim=16)
    model = TgRN(config).eval()
    text, audio, vision = torch.randn(2, 5, 4), torch.randn(2, 4, 3), torch.randn(2, 6, 2)
    output = model(
        text, audio, vision,
        text_valid=torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
        audio_valid=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool),
        vision_valid=torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool),
    )
    assert torch.isfinite(output.prediction).all()
