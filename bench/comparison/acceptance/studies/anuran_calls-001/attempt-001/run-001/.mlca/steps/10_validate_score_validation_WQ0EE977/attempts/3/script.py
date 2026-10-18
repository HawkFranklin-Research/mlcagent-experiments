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

feature_cols = [c for c in df_train.columns
                if c not in targets and c not in excluded and c != group_col]

def load_pkl(k):
    with open(inp[k], "rb") as f:
        return pickle.load(f)

pre = load_pkl("06_preprocess_fit_preprocessing_4H3435T3/preprocessor.pkl")
selector = load_pkl("07_features_select_features_D255Z34M/selector.pkl")
model_a = load_pkl("08_model_train_logistic_YE83DHCX/model.pkl")
model_b = load_pkl("09_model_train_alternative_0RK3GGA1/alternative_model.pkl")

def wrap(m):
    if isinstance(m, Pipeline):
        return m
    if isinstance(selector, Pipeline):
        return Pipeline([("sel", selector), ("model", m)])
    return Pipeline([("pre", pre), ("sel", selector), ("model", m)])

candidates = {"08_logistic": wrap(model_a), "09_alternative": wrap(model_b)}
X_val = df_val[feature_cols]
y_val = df_val[targets[0]].to_numpy()

rows, val_scores = [], {}
for name, pipe in candidates.items():
    first = list(pipe.named_steps)[0]
    expected = getattr(pipe.named_steps[first], "feature_names_in_", None)
    Xv = X_val[list(expected)] if expected is not None else X_val
    classes = np.asarray(pipe.classes_)
    proba = pipe.predict_proba(Xv)
    pred = pipe.predict(Xv)
    mask = np.isin(y_val, classes)
    y_use, p_use, pred_use = y_val[mask], proba[mask], pred[mask]
    present = np.array([c for c in classes if (y_use == c).any()])
    if len(present) < 2:
        raise ValueError(f"{name}: fewer than 2 val classes overlapping model classes")
    col_idx = [int(np.where(classes == c)[0][0]) for c in present]
    auc = roc_auc_score(y_use, p_use[:, col_idx], multi_class="ovr",
                        average="macro", labels=present)
    f1 = f1_score(y_use, pred_use, average="macro", labels=present)
    if not np.isfinite(auc):
        raise ValueError(f"{name}: non-finite roc_auc_ovr")
    rows.append({"model": name, "metric": primary, "value": float(auc), "threshold": np.nan})
    rows.append({"model": name, "metric": "f1_macro", "value": float(f1), "threshold": np.nan})
    val_scores[name] = float(auc)
    mlca_runtime.record_metric(primary, float(auc), "val")

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "validation_metrics.csv",
                        caption="Validation metrics per candidate model")

best = max(val_scores, key=val_scores.get)
mlca_runtime.record_fact_candidate("best_model", best, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_val_roc_auc", val_scores[best],
                                   source="validation_metrics.csv")

print(f"features={len(feature_cols)} n_classes={task['n_classes']}")
print(f"primary={primary} best={best} val_auc={val_scores[best]:.4f}")
