from __future__ import annotations

import numpy as np


def _weighted_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Match sklearn's binary ``average='weighted'`` F1 without a dependency."""
    scores = []
    supports = []
    for label in (False, True):
        actual = truth == label
        predicted = prediction == label
        tp = int(np.sum(actual & predicted))
        fp = int(np.sum(~actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
        supports.append(int(np.sum(actual)))
    total = sum(supports)
    return float(np.average(scores, weights=supports)) if total else 0.0


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """CMU-MOSI metrics used by TgRN and its public TFR-Net parent code.

    MultAcc-5/7 round clipped continuous predictions. Binary F1 is support-
    weighted across both classes; it is not positive-class-only F1.
    """
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same number of elements")
    if not (np.isfinite(prediction).all() and np.isfinite(target).all()):
        raise ValueError("prediction and target must contain only finite values")
    mae = float(np.mean(np.abs(prediction - target)))
    if len(target) > 1 and np.std(prediction) > 0 and np.std(target) > 0:
        corr = float(np.corrcoef(prediction, target)[0, 1])
    else:
        corr = 0.0

    mult_acc7 = float(np.mean(np.round(np.clip(prediction, -3, 3)) == np.round(np.clip(target, -3, 3))))
    mult_acc5 = float(np.mean(np.round(np.clip(prediction, -2, 2)) == np.round(np.clip(target, -2, 2))))

    def binary(include_zero: bool) -> tuple[float, float]:
        keep = np.ones_like(target, dtype=bool) if include_zero else target != 0
        truth = target[keep] >= 0 if include_zero else target[keep] > 0
        pred = prediction[keep] >= 0 if include_zero else prediction[keep] > 0
        acc = float(np.mean(truth == pred)) if len(truth) else 0.0
        return acc, _weighted_f1(truth, pred)

    has0_acc2, has0_f1 = binary(True)
    non0_acc2, non0_f1 = binary(False)
    return {
        "mae": mae,
        "corr": corr,
        "mult_acc5": mult_acc5,
        "mult_acc7": mult_acc7,
        "has0_acc2": has0_acc2,
        "has0_f1": has0_f1,
        "non0_acc2": non0_acc2,
        "non0_f1": non0_f1,
    }
