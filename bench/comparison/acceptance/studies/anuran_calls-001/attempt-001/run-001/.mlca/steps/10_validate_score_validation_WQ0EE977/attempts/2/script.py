import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
primary = task["primary_metric"]

df_train = pd.read_parquet(inp["train"])
df_val = pd.read_parquet(inp["val"])

feature_cols = []
for c in df_train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if df_train[c].dtype == object and df_train[c].nunique() == len(df_train):
        continue
    if df_train[c].dtype.kind in "iu" and df_train[c].nunique() == len(df_train):
        continue
    feature_cols.append(c)

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("06_preprocess_fit_preprocessing_4H3435T3/preprocessor.pkl")
selector = load_pkl("07_features_select_features_D255Z34M/selector.pkl")
model_a = load_pkl("08_model_train_logistic_YE83DHCX/model.pkl")
model_b = load_pkl("09_model_train_alternative_0RK3GGA1/alternative_model.pkl")

def make_complete(m):
    if isinstance(m, Pipeline):
        return m
    if isinstance(selector, Pipeline):
        return Pipeline([("sel", selector), ("model", m)])
    return Pipeline([("pre", pre), ("sel", selector), ("model", m)])

candidates = {
    "08_logistic": make_complete(model_a),
    "09_alternative": make_complete(model_b),
}

X_val = df_val[feature_cols]
y_val = df_val[targets[0]].to_numpy()
val_counts = pd.Series(y_val).value_counts()

rows = []
val_scores = {}
for name, pipe in candidates.items():
    first_name = list(pipe.named_steps.keys())[0]
    expected = getattr(pipe.named_steps[first_name], "feature_names_in_", None)
    Xv = X_val[list(expected)] if expected is not None else X_val

    classes = getattr(pipe, "classes_", None)
    if classes is None:
        classes = pipe.named_steps[list(pipe.named_steps)[-1]].classes_
    classes = np.asarray(classes)

    proba = pipe.predict_proba(Xv)
    pred = pipe.predict(Xv)

    present = [c for c in classes if c in val_counts.index and 0 < val_counts[c] < len(y_val)]
    if len(present) == 0:
        present = [c for c in classes if c in val_counts.index]
    if len(present) == 0:
        raise ValueError(
            f"no overlap between model classes {classes.tolist()} "
            f"and val labels {sorted(val_counts.index.tolist())}"
        )
    present = np.asarray(present)
    col_idx = [int(np.where(classes == c)[0][0]) for c in present]

    if len(present) >= 2:
        auc = roc_auc_score(
            y_val, proba[:, col_idx],
            multi_class="ovr", average="macro", labels=present,
        )
    else:
        auc = float("nan")
    f1 = f1_score(y_val, pred, average="macro", labels=present)

    if not np.isfinite(auc):
        raise ValueError(
            f"roc_auc_ovr non-finite for {name}: present={present.tolist()}, "
            f"val_counts={val_counts.loc[present].to_dict()}"
        )

    rows.append({"model": name, "metric": primary, "value": float(auc), "threshold": np.nan})
    rows.append({"model": name, "metric": "roc_auc_ovr_supp", "value": float(auc), "threshold": np.nan})
    rows.append({"model": name, "metric": "f1_macro", "value": float(f1), "threshold": np.nan})
    val_scores[name] = float(auc)

metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(
    metrics_df, "validation_metrics.csv",
    caption="Validation metrics per candidate model",
)

best_name = max(val_scores, key=val_scores.get)
best_val = val_scores[best_name]
for name, score in val_scores.items():
    mlca_runtime.record_metric(primary, float(score), "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", best_val, source="validation_metrics.csv")

print(f"features={len(feature_cols)} n_classes={task['n_classes']}")
print(f"primary={primary}")
print(f"best_model={best_name} val_auc={best_val:.4f}")
