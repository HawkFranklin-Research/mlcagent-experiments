"""Identical prediction scoring for the agent and both AutoML comparators."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def score(
    problem_type, metric, y_true, y_pred=None, proba=None, classes=None, positive=None
):
    truth = np.asarray(y_true)
    if truth.ndim != 1 or len(truth) == 0:
        raise ValueError("Expected nonempty single-target labels")
    notes, secondary = [], {}
    if problem_type == "regression":
        if metric != "rmse":
            raise ValueError(f"Unsupported regression metric: {metric}")
        value = np.sqrt(mean_squared_error(truth, y_pred))
        secondary = {
            "mae": mean_absolute_error(truth, y_pred),
            "r2": r2_score(truth, y_pred),
        }
    elif problem_type in {"binary", "multiclass"}:
        classes = (
            list(classes) if classes is not None else sorted(np.unique(truth).tolist())
        )
        if len(set(classes)) != len(classes) or not set(truth).issubset(set(classes)):
            raise ValueError("Invalid or unknown class labels")
        probabilities = np.asarray(proba, dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Invalid probabilities")
        if problem_type == "binary" and metric == "roc_auc":
            if len(classes) != 2:
                raise ValueError("Binary scoring requires two training classes")
            positive = sorted(classes)[1] if positive is None else positive
            if positive not in classes:
                raise ValueError("Positive label is not a training class")
            if probabilities.ndim == 2:
                if probabilities.shape != (len(truth), 2):
                    raise ValueError("Probability matrix shape mismatch")
                if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
                    raise ValueError("Binary probabilities must sum to one")
                probabilities = probabilities[:, classes.index(positive)]
            if probabilities.shape != (len(truth),):
                raise ValueError("Probability vector shape mismatch")
            indicator = truth == positive
            if len(np.unique(indicator)) < 2:
                value = None
                notes.append("AUC unavailable: evaluation rows contain only one class")
            else:
                value = roc_auc_score(indicator, probabilities)
        elif problem_type == "multiclass" and metric == "roc_auc_ovr":
            if probabilities.shape != (len(truth), len(classes)):
                raise ValueError("Probability matrix shape mismatch")
            if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
                raise ValueError("Multiclass probabilities must sum to one")
            aucs, omitted = [], []
            for index, cls in enumerate(classes):
                indicator = truth == cls
                if len(np.unique(indicator)) < 2:
                    omitted.append(cls)
                else:
                    aucs.append(roc_auc_score(indicator, probabilities[:, index]))
            value = float(np.mean(aucs)) if aucs else None
            if omitted:
                notes.append({"omitted_auc_classes": omitted})
        else:
            raise ValueError(f"Unsupported classification metric: {metric}")
        if y_pred is not None:
            secondary = {
                "balanced_accuracy": balanced_accuracy_score(truth, y_pred),
                "macro_f1": f1_score(truth, y_pred, average="macro", zero_division=0),
                "accuracy": accuracy_score(truth, y_pred),
            }
    else:
        raise ValueError("Unsupported problem type")

    def finite(number):
        return float(number) if number is not None and np.isfinite(number) else None

    return {
        "primary": {"name": metric, "value": finite(value)},
        "secondary": {k: finite(v) for k, v in secondary.items()},
        "n": len(truth),
        "notes": notes,
    }
