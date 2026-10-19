import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def keylike(s):
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        v = s.dropna().sort_values().to_numpy()
        return len(v) == len(s) and len(v) > 1 and np.all(np.diff(v) == 1)
    return False

excl = set(targets) | set(excluded)
if group_col:
    excl.add(group_col)
features = [c for c in train.columns if c not in excl and not keylike(train[c])]

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

pre = load(inp["05_preprocess_fit_preprocessing_5G7E554W/preprocessor.pkl"])
sel = load(inp["06_features_select_features_Y1ATFPQ6/selector.pkl"])
mdl = load(inp["latest_model"])

def raw_pipe(o):
    if isinstance(o, Pipeline) and hasattr(o, "feature_names_in_"):
        return list(o.feature_names_in_) == list(features)
    return False

def has_select(o):
    return isinstance(o, Pipeline) and any("select" in n.lower() for n in o.named_steps)

if raw_pipe(mdl):
    full = mdl
elif raw_pipe(sel):
    full = Pipeline([("features", sel), ("model", mdl)])
elif isinstance(pre, Pipeline) and has_select(pre):
    full = Pipeline([("features", pre), ("model", mdl)])
else:
    full = Pipeline([("preprocessor", pre), ("selector", sel), ("model", mdl)])

expected = getattr(full, "feature_names_in_", None)
if expected is not None:
    expected = list(expected)
    bad = set(expected) & (set(targets) | set(excluded) | ({group_col} if group_col else set()))
    if bad:
        raise ValueError(f"Pipeline expects excluded columns: {bad}")
    features = expected

X_val = val[features].copy()
y_val = val[targets[0]].copy()
scoring = "roc_auc" if task.get("type") == "binary" or task.get("n_classes") == 2 else "roc_auc_ovr"
if primary in ("roc_auc", "roc_auc_ovr", "roc_auc_ovo"):
    scoring = primary

r = permutation_importance(
    full, X_val, y_val, scoring=scoring, n_repeats=20, random_state=42, n_jobs=1
)
df = pd.DataFrame({
    "feature": features,
    "importance_mean": r.importances_mean,
    "importance_sd": r.importances_std,
})
df = df.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

def sig4(x):
    return float(f"{x:.4g}") if pd.notna(x) else x

df["importance_mean"] = df["importance_mean"].map(sig4)
df["importance_sd"] = df["importance_sd"].map(sig4)
mlca_runtime.save_table(df, "interpretation.csv", caption="Permutation importance on validation rows")
top = df.iloc[0]["feature"]
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print(f"features: {len(features)}")
print(f"val rows: {len(val)}")
print(f"top feature: {top}")
