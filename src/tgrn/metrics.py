from __future__ import annotations

import numpy as np


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    mae = float(np.mean(np.abs(prediction - target)))
    corr = float(np.corrcoef(prediction, target)[0, 1]) if len(target) > 1 else 0.0

    def binary(include_zero: bool) -> tuple[float, float]:
        keep = np.ones_like(target, dtype=bool) if include_zero else target != 0
        truth = target[keep] >= 0 if include_zero else target[keep] > 0
        pred = prediction[keep] >= 0 if include_zero else prediction[keep] > 0
        acc = float(np.mean(truth == pred)) if len(truth) else 0.0
        tp = np.sum(truth & pred)
        fp = np.sum(~truth & pred)
        fn = np.sum(truth & ~pred)
        f1 = float(2 * tp / max(2 * tp + fp + fn, 1))
        return acc, f1

    has0_acc2, has0_f1 = binary(True)
    non0_acc2, non0_f1 = binary(False)
    return {
        "mae": mae,
        "corr": corr,
        "has0_acc2": has0_acc2,
        "has0_f1": has0_f1,
        "non0_acc2": non0_acc2,
        "non0_f1": non0_f1,
    }
