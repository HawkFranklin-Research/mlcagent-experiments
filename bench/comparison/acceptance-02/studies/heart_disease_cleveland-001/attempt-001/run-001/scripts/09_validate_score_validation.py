import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_curve,
    f1_score,
)


def sig4(x):
    if x is None:
        return np.nan
    try:
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return 0.0 if x == 0 else float(f"{x:.4g}")
    except Exception:
        return np.nan


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_classes(est):
    if hasattr(est, "classes_"):
        return np.asarray(est.classes_)
    if hasattr(est, "named_steps"):
        final = est.named_steps[list(est.named_steps)[-1]]
        return np.asarray(final.classes_)
    raise AttributeError("Model has no classes_")


def assemble(model, pre, sel):
    if isinstance(model, Pipeline):
        return model
    if isinstance(sel, Pipeline):
        return Pipeline([("selector", sel), ("model", model)])
    return Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])


def align(est, df):
    if hasattr(est, "feature_names_in_"):
        cols = [str(c) for c in est.feature_names_in_]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing input feature columns: {missing}")
        return df.loc[:, cols]
    return df


def binary_metrics(est, X, y, task):
    classes = get_classes(est)
    pos = task.get("positive_label", classes[1])
    if pos not in classes:
        pos = classes[1]
    j = int(np.where(classes == pos)[0][0])
    p = est.predict_proba(X)[:, j]
    yb = (y == pos).astype(int).to_numpy() if hasattr(y, "to_numpy") else (y == pos).astype(int)
    auc = roc_auc_score(yb, p)
    ap = average_precision_score(yb, p)
    fpr, tpr, thr = roc_curve(yb, p)
    fin = np.isfinite(thr)
    youden = tpr[fin] - fpr[fin]
    cut = float(thr[fin][int(np.argmax(youden))])
    pred = (p >= cut).astype(int)
    bal = balanced_accuracy_score(yb, pred)
    brier = brier_score_loss(yb, p)
    return [
        ("roc_auc", auc, np.nan),
        ("pr_auc", ap, np.nan),
        ("balanced_accuracy", bal, cut),
        ("brier_score", brier, np.nan),
    ], float(auc)


def multiclass_metrics(est, X, y, task):
    classes = get_classes(est)
    proba = est.predict_proba(X)
    present = [c for c in classes if (y == c).any()]
    if len(present) == len(classes):
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
    else:
        aucs = []
        for c in present:
            j = int(np.where(classes == c)[0][0])
            aucs.append(roc_auc_score((y == c).astype(int), proba[:, j]))
        auc = float(np.mean(aucs))
        print("classes omitted from roc_auc:", [c for c in classes if c not in present])
    pred = est.predict(X)
    f1 = f1_score(y, pred, average="macro", labels=classes)
    auc_name = "roc_auc_ovr" if task.get("primary_metric") == "roc_auc_ovr" else "roc_auc"
    return [(auc_name, auc, np.nan), ("f1_macro", f1, np.nan)], float(auc)


def main():
    mlca_runtime.set_seed(42)
    inp = mlca_runtime.inputs()
    protocol = json.loads(inp["_protocol"].read_text())

    try:
        task = protocol["task"]
        targets = protocol["targets"]
        excluded = protocol["excluded_columns"]
    except KeyError as e:
        raise KeyError(
            f"Missing protocol key {e}; available top-level keys: {list(protocol.keys())}"
        ) from e

    group_col = protocol.get("group_column")
    train = pd.read_parquet(inp["train"])
    val = pd.read_parquet(inp["val"])
    target = targets[0]

    feature_cols = [
        c
        for c in train.columns
        if c not in set(targets) | set(excluded) and c != group_col
    ]
    y_val = val[target]

    pre = load_pkl(inp["05_preprocess_fit_preprocessing_5G7E554W/preprocessor.pkl"])
    sel = load_pkl(inp["06_features_select_features_Y1ATFPQ6/selector.pkl"])

    candidates = [
        ("logistic", inp["07_model_train_logistic_12XC5SW7/model.pkl"]),
        ("alternative", inp["08_model_train_alternative_0NB2T98C/alternative_model.pkl"]),
    ]

    primary = task["primary_metric"]
    lower_is_better = primary in {"brier_score", "log_loss", "mae", "mse", "rmse"}
    rows, info = [], []

    for name, path in candidates:
        model = load_pkl(path)
        est = assemble(model, pre, sel)
        Xv = align(est, val.loc[:, feature_cols])

        if task.get("type") == "binary":
            metrics, roc_val = binary_metrics(est, Xv, y_val, task)
        else:
            metrics, roc_val = multiclass_metrics(est, Xv, y_val, task)

        for metric, value, threshold in metrics:
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "value": sig4(value),
                    "threshold": sig4(threshold),
                }
            )

        try:
            primary_value = next(v for m, v, _ in metrics if m == primary)
        except StopIteration:
            raise KeyError(
                f"Primary metric {primary} not computed; available: {[m for m, _, _ in metrics]}"
            )

        info.append(
            {
                "model": name,
                "primary": float(primary_value),
                "roc_auc": float(roc_val),
            }
        )
        mlca_runtime.record_metric(primary, float(primary_value), split="val")

    best = min(info, key=lambda d: d["primary"]) if lower_is_better else max(info, key=lambda d: d["primary"])
    mlca_runtime.record_metric(primary, best["primary"], split="val")
    mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")
    mlca_runtime.record_fact_candidate("best_val_roc_auc", best["roc_auc"], source="validation_metrics.csv")

    mlca_runtime.save_table(
        pd.DataFrame(rows, columns=["model", "metric", "value", "threshold"]),
        "validation_metrics.csv",
        caption="Validation metrics by candidate model.",
    )
    print(f"rows train/val: {len(train)}/{len(val)}; candidates: {len(candidates)}")
    print(f"best_model: {best['model']}; best_val_roc_auc: {sig4(best['roc_auc'])}")


if __name__ == "__main__":
    main()
