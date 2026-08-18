from __future__ import annotations

import numpy as np
from sklearn import metrics as sk_metrics


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float | int | None]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    prediction = (p >= threshold).astype(int)
    matrix = sk_metrics.confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())

    def ratio(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator else None

    result: dict[str, float | int | None] = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()) if len(y) else None,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": ratio(tn + tp, len(y)),
        "precision_non_event": ratio(tn, tn + fn),
        "precision_event": ratio(tp, tp + fp),
        "recall_non_event": ratio(tn, tn + fp),
        "recall_event": ratio(tp, tp + fn),
    }
    if np.unique(y).size < 2:
        result.update({"auroc": None, "auprc": None, "minpse": None})
        return result
    precision, recall, _ = sk_metrics.precision_recall_curve(y, p)
    result.update(
        {
            "auroc": float(sk_metrics.roc_auc_score(y, p)),
            "auprc": float(sk_metrics.auc(recall, precision)),
            "minpse": float(max(min(x, z) for x, z in zip(precision, recall))),
        }
    )
    return result

