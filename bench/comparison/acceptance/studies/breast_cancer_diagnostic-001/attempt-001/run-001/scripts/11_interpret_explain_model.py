import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

SEED = 42
mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

try:
    targets = list(protocol["targets"])
    excluded = list(protocol.get("excluded_columns", []))
    group = protocol.get("group_column")
    metric = protocol["task"]["primary_metric"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; top-level keys: {list(protocol.keys())}") from exc

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def key_like(s: pd.Series) -> bool:
    n = int(s.notna().sum())
    if n <= 1:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return int(s.nunique(dropna=True)) == n
    if pd.api.types.is_integer_dtype(s):
        return int(s.nunique(dropna=True)) == n
    return False

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_pickle(inp["07_model_train_logistic_BMNCC8GS/model.pkl"])
pre = load_pickle(inp["05_preprocess_fit_preprocessing_WZN8YG48/preprocessor.pkl"])
sel = load_pickle(inp["06_features_select_features_6Y9H969T/selector.pkl"])

if isinstance(model, Pipeline) and len(model.named_steps) > 1:
    pipe = model
elif isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
    pipe = Pipeline([("selector", sel), ("model", model)])
else:
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

input_cols = list(getattr(pipe, "feature_names_in_", feature_cols))
missing = [c for c in input_cols if c not in val.columns]
if missing:
    raise KeyError(f"Validation missing model input columns: {missing}")

X_val = val.loc[:, input_cols].copy()
target = targets[0]
y_val = val[target]

scoring = "roc_auc" if str(metric).lower() in ("roc_auc", "auc") else metric
res = permutation_importance(
    pipe, X_val, y_val, scoring=scoring, n_repeats=20, random_state=SEED, n_jobs=1
)

imp = pd.DataFrame({
    "feature": X_val.columns,
    "importance_mean": res.importances_mean,
    "importance_sd": res.importances_std,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for c in ["importance_mean", "importance_sd"]:
    imp[c] = imp[c].map(lambda v: float(f"{float(v):.4g}"))

mlca_runtime.save_table(imp, "interpretation.csv", caption="Validation permutation importance (20 repeats)")
if len(imp):
    mlca_runtime.record_fact_candidate(
        "top_feature", str(imp.loc[0, "feature"]), source="interpretation.csv"
    )

print(f"rows={len(val)} features={len(input_cols)} repeats=20")
print(f"top_feature={imp.loc[0, 'feature'] if len(imp) else 'none'}")
