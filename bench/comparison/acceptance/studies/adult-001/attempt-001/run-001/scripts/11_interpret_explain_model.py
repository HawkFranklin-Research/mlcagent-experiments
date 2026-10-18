import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

SEED = 42
mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {sorted(d)}")
    return d[k]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
task = need(protocol, "task")
metric = need(task, "primary_metric")
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
forbidden = set(targets) | set(excluded) | ({group} if group else set())

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        n = str(s.name).lower()
        return s.nunique(dropna=True) == len(s) and any(t in n for t in ("id", "idx", "index", "key", "row"))
    return False

feature_cols = [c for c in train.columns if c not in forbidden and not key_like(train[c])]

def load(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

cands = []
for k in ("latest_model", "07_model_train_logistic_QY4T2JSR/model.pkl", "08_model_train_alternative_5C3RWNJ3/alternative_model.pkl"):
    try:
        cands.append(load(k))
    except Exception:
        pass

pipes = [
    o for o in cands
    if isinstance(o, Pipeline)
    and hasattr(o, "feature_names_in_")
    and (hasattr(o, "predict_proba") or hasattr(o, "decision_function"))
]
if pipes:
    best = max(pipes, key=lambda o: len(o.named_steps))
else:
    pre = load("05_preprocess_fit_preprocessing_XT8NJ07Q/preprocessor.pkl")
    sel = load("06_features_select_features_A3NFBXN9/selector.pkl")
    mdl = load("07_model_train_logistic_QY4T2JSR/model.pkl")
    best = Pipeline([("preprocessor", pre), ("selector", sel), ("model", mdl)]) if sel is not None else Pipeline([("preprocessor", pre), ("model", mdl)])

input_cols = list(best.feature_names_in_) if hasattr(best, "feature_names_in_") else feature_cols
if set(input_cols) & forbidden:
    raise ValueError("Target/excluded/group column reached the pipeline input")
if not set(input_cols).issubset(set(train.columns)):
    raise ValueError("Pipeline expects columns absent from the frozen table")

X_val = val[input_cols].copy()
y_val = val[target]
scoring = "roc_auc" if task.get("n_classes", 2) == 2 or metric in ("roc_auc", "auc") else "roc_auc_ovr"
res = permutation_importance(best, X_val, y_val, scoring=scoring, n_repeats=20, random_state=SEED, n_jobs=1)
out = pd.DataFrame({
    "feature": input_cols,
    "importance_mean": res.importances_mean,
    "importance_sd": res.importances_std,
})
out = out.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for c in ("importance_mean", "importance_sd"):
    out[c] = out[c].map(lambda x: float(f"{x:.4g}"))

mlca_runtime.save_table(out, "interpretation.csv", caption="Permutation importance on validation rows")
top_feature = str(out.loc[0, "feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"features={len(input_cols)} val_rows={len(X_val)}")
print(f"top_feature={top_feature}")
