import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

p = json.loads(inp["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in p:
        raise KeyError(f"Missing _protocol key {k!r}; available keys: {list(p)}")
targets = p["targets"]
excluded = list(p["excluded_columns"])
group = p.get("group_column")
task = p["task"]
metric = task.get("primary_metric", "roc_auc")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def feature_columns(df):
    drop = set(targets) | set(excluded) | ({group} if group else set())
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or
                pd.api.types.is_integer_dtype(s)):
            if len(s) > 1 and s.nunique(dropna=False) == len(s):
                continue
        cols.append(c)
    return cols

feature_cols = feature_columns(val)
with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)

input_cols = list(getattr(model, "feature_names_in_", feature_cols))
input_cols = [c for c in input_cols if c in val.columns]
X = val[input_cols].copy()
target = targets[0]
y = val[target]

def score_model(Xp):
    proba = model.predict_proba(Xp)
    cls = getattr(model, "classes_", None)
    if proba.shape[1] == 2:
        pos = cls[1] if cls is not None else 1
        y_bin = (y == pos).astype(int)
        return roc_auc_score(y_bin, proba[:, 1])
    return roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=cls)

out_cols = ["feature", "importance_mean", "importance_sd"]
if len(input_cols) > 500:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    imp = None
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    elif hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_)
    if imp is None or len(imp) != len(input_cols):
        interp = pd.DataFrame(columns=out_cols)
        mlca_runtime.save_table(interp, "interpretation.csv", "Permutation importance unavailable")
        mlca_runtime.record_fact_candidate("top_feature", "unavailable", source="interpretation.csv")
        print(f"val rows={len(val)} features={len(input_cols)}")
        print("reliable importance ranking unavailable")
        raise SystemExit(0)
    keep = np.argsort(imp)[::-1][:50]
    perm_cols = [input_cols[i] for i in keep]
    n_repeats = 5
else:
    perm_cols = input_cols
    n_repeats = 20

base = score_model(X)
rng = np.random.default_rng(42)
rows = []
for col in perm_cols:
    vals = []
    for _ in range(n_repeats):
        Xp = X.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        vals.append(base - score_model(Xp))
    rows.append((col, float(np.mean(vals)), float(np.std(vals))))

interp = pd.DataFrame(rows, columns=out_cols).sort_values("importance_mean", ascending=False).head(15)
for c in ["importance_mean", "importance_sd"]:
    interp[c] = interp[c].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(interp, "interpretation.csv", "Validation permutation importance")

top = interp.iloc[0]["feature"] if len(interp) else "unavailable"
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print(f"val rows={len(val)} train rows={len(train)} features={len(input_cols)}")
print(f"permuted={len(perm_cols)} repeats={n_repeats}")
print(f"top_feature={top}")
