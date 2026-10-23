import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for key in ("task", "targets"):
    if key not in proto:
        raise ValueError(f"missing protocol key {key}; available={list(proto.keys())}")
task = proto["task"]
if "primary_metric" not in task:
    raise ValueError(f"task missing primary_metric; available={list(proto.keys())}")
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column")
primary = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
missing = [c for c in targets if c not in train.columns or c not in val.columns]
if missing:
    raise ValueError(f"target columns missing from train/val: {missing}")
target = targets[0]

def is_key_like(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        nm = str(s.name).lower()
        return any(t in nm for t in ("id", "key", "code")) and s.nunique(dropna=True) == len(s)
    return False

excl = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in excl and not is_key_like(train[c])]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_M7AXCKT1/preprocessor.pkl"])
sel = load_pkl(inp["06_features_select_features_XFCK1C59/selector.pkl"])

def make_predictor(model):
    if isinstance(model, Pipeline):
        return model
    n = getattr(model, "n_features_in_", None)
    if n is None or n == len(feature_cols):
        return model
    if sel is not None:
        if isinstance(sel, Pipeline):
            sc = getattr(sel, "feature_names_in_", None)
            if sc is not None and set(sc).issubset(set(feature_cols)):
                front = sel
            elif pre is not None:
                front = Pipeline([("preprocessor", pre), ("selector", sel)])
            else:
                front = sel
        elif pre is not None:
            front = Pipeline([("preprocessor", pre), ("selector", sel)])
        else:
            front = sel
        return Pipeline([("frontend", front), ("model", model)])
    if pre is not None:
        return Pipeline([("preprocessor", pre), ("model", model)])
    return model

models = [
    ("logistic", load_pkl(inp["07_model_train_logistic_YKXH5QEQ/model.pkl"])),
    ("alternative", load_pkl(inp["08_model_train_alternative_6PKR6EV7/alternative_model.pkl"])),
]

rows = []
for name, raw_model in models:
    model = make_predictor(raw_model)
    cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feature_cols
    miss = [c for c in cols if c not in val.columns]
    if miss:
        raise ValueError(f"{name} missing columns; first={miss[0]}")
    Xv = val.loc[:, cols]
    proba = model.predict_proba(Xv)
    classes = np.asarray(model.classes_)
    yv = val[target].to_numpy()

    if len(classes) == 2:
        pos_label = task.get("positive_label")
        if pos_label is None or pos_label not in classes:
            pos_label = classes[1]
        pos_idx = int(np.where(classes == pos_label)[0][0])
        p = np.asarray(proba)[:, pos_idx]
        y_bin = (yv == pos_label).astype(int)
        roc = roc_auc_score(y_bin, p)
        pr = average_precision_score(y_bin, p)
        fpr, tpr, thr = roc_curve(y_bin, p)
        j = tpr - fpr
        idx = int(np.argmax(j[1:]) + 1) if len(j) > 1 else 0
        youden = float(thr[idx])
        pred = (p >= youden).astype(int)
        bal = balanced_accuracy_score(y_bin, pred)
        brier = brier_score_loss(y_bin, p)
        for metric, value in {
            "roc_auc": roc,
            "pr_auc": pr,
            "balanced_accuracy": bal,
            "brier": brier,
        }.items():
            rows.append({
                "model": name,
                "metric": metric,
                "value": float(value),
                "threshold": youden if metric == "balanced_accuracy" else np.nan,
            })
    else:
        pred = classes[np.argmax(proba, axis=1)]
        f1 = f1_score(yv, pred, average="macro")
        present = [c for c in classes if np.any(yv == c)]
        if len(present) == len(classes):
            roc = roc_auc_score(yv, proba, multi_class="ovr", average="macro", labels=classes)
        else:
            aucs = []
            for c in present:
                j = int(np.where(classes == c)[0][0])
                aucs.append(roc_auc_score((yv == c).astype(int), np.asarray(proba)[:, j]))
            roc = float(np.mean(aucs))
        for metric, value in {"roc_auc_ovr": roc, "f1_macro": f1}.items():
            rows.append({
                "model": name,
                "metric": metric,
                "value": float(value),
                "threshold": np.nan,
            })

metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].map(lambda x: float(f"{x:.4g}"))
metrics_df["threshold"] = metrics_df["threshold"].map(lambda x: np.nan if pd.isna(x) else float(f"{x:.4g}"))
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation metrics by candidate model")

if primary not in set(metrics_df["metric"]):
    raise ValueError(f"primary metric {primary} not computed; available={sorted(set(metrics_df['metric']))}")
sub = metrics_df[metrics_df["metric"] == primary]
higher = primary in {"roc_auc", "roc_auc_ovr", "pr_auc", "balanced_accuracy", "f1_macro", "accuracy"}
best_row = sub.loc[sub["value"].idxmax() if higher else sub["value"].idxmin()]
best_model = str(best_row["model"])
best_primary = float(best_row["value"])

auc_name = "roc_auc" if "roc_auc" in set(metrics_df["metric"]) else "roc_auc_ovr"
auc_sub = metrics_df[(metrics_df["model"] == best_model) & (metrics_df["metric"] == auc_name)]
if auc_sub.empty:
    raise ValueError("best_val_roc_auc not available")
best_auc = float(auc_sub.iloc[0]["value"])

mlca_runtime.record_metric(primary, best_primary, split="val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_auc, source="validation_metrics.csv")

print(f"train {train.shape}, val {val.shape}, features {len(feature_cols)}")
print(f"best {best_model}: {primary}={best_primary:.4g}")
print(f"best_val_roc_auc={best_auc:.4g}")
