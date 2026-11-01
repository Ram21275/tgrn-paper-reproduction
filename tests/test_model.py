import torch
import pickle
import tempfile
from pathlib import Path

import numpy as np

from tgrn import TgRN, TgRNConfig
from tgrn.data import MultimodalPickleDataset
from tgrn.masking import apply_random_missingness
from tgrn.metrics import regression_metrics


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


def test_dataset_uses_bert_attention_mask() -> None:
    split = {
        "text": np.ones((2, 4, 6), dtype=np.float32),
        "text_bert": np.array(
            [
                [[101, 10, 102, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
                [[101, 11, 12, 102], [1, 1, 1, 1], [0, 0, 0, 0]],
            ],
            dtype=np.int64,
        ),
        "audio": np.ones((2, 3, 2), dtype=np.float32),
        "vision": np.ones((2, 5, 3), dtype=np.float32),
        "audio_lengths": np.array([2, 3]),
        "vision_lengths": np.array([4, 5]),
        "regression_labels": np.array([-1.0, 1.0], dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "features.pkl"
        with path.open("wb") as stream:
            pickle.dump({"train": split}, stream)
        sample = MultimodalPickleDataset(path, "train")[0]
    assert sample["text_valid"].tolist() == [True, True, True, False]
    assert sample["audio_valid"].tolist() == [True, True, False]
    assert sample["vision_valid"].tolist() == [True, True, True, True, False]


def test_mosi_metrics_match_reference_definitions() -> None:
    target = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    prediction = np.array([-2.6, 0.2, -0.1, 0.8, 2.7])
    metrics = regression_metrics(prediction, target)
    assert metrics["mult_acc7"] == 0.8
    assert metrics["mult_acc5"] == 0.8
    assert metrics["has0_acc2"] == 0.6
    assert metrics["non0_acc2"] == 0.75
    # Weighted F1 differs from the positive-class-only score used previously.
    assert np.isclose(metrics["has0_f1"], 0.6)
    assert np.isclose(metrics["non0_f1"], 0.7333333333333334)
