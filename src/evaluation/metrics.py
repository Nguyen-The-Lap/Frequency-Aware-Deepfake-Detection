import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


def _eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def evaluate_predictions(labels, scores, threshold=0.5):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be non-empty arrays of equal shape")
    pred = (scores >= threshold).astype(int)
    result = {"accuracy": float(accuracy_score(labels, pred)),
              "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist()}
    result["roc_auc"] = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")
    result["eer"] = _eer(labels, scores) if np.unique(labels).size == 2 else float("nan")
    return result


def aggregate_video_predictions(video_ids, labels, scores):
    """Mean frame score per video; rejects mixed ground-truth labels within a video."""
    import pandas as pd
    frame = pd.DataFrame({"video_id": video_ids, "label": labels, "score": scores})
    grouped = frame.groupby("video_id", sort=True)
    if (grouped.label.nunique() > 1).any():
        raise ValueError("A video has inconsistent ground-truth labels")
    return grouped.agg(label=("label", "first"), score=("score", "mean")).reset_index()
