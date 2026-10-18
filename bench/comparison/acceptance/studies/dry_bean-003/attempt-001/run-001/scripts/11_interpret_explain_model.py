import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

SEED = 42
mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
metric = task["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

def key_like(s):
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s) and any(k in s.name.lower() for k in ("id", "key", "index"))
    return False

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]
if target in feature_cols:
    raise ValueError("Target leaked into feature columns")

def load_pkl(k):
    return pickle.loads(inp[k].read_bytes())

pipe = None
for k in ["latest_model", "08_model_train_alternative_M33R6NHM/alternative_model.pkl", "07_model_train_logistic_KTFJXM94/model.pkl"]:
    obj = load_pkl(k)
    if isinstance(obj, Pipeline) and hasattr(obj, "predict_proba"):
        pipe = obj
        break
if pipe is None:
    raise TypeError("No complete saved Pipeline with predict_proba found")

use_cols = list(pipe.feature_names_in_) if hasattr(pipe, "feature_names_in_") else feature_cols
bad = set(use_cols) & (set(targets) | set(excluded) | ({group} if group else set()))
if bad:
    raise ValueError(f"Target/excluded columns in pipeline input: {sorted(bad)}")
missing = [c for c in use_cols if c not in val.columns]
if missing:
    raise ValueError(f"Missing validation columns: {missing}")

X_val = val[use_cols].copy()
y_val = val[target].copy()

pi = permutation_importance(
    pipe, X_val, y_val, scoring=metric, n_repeats=20,
    random_state=SEED, n_jobs=1,
)
df = pd.DataFrame({
    "feature": use_cols,
    "importance_mean": pi.importances_mean,
    "importance_sd": pi.importances_std,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

for c in ["importance_mean", "importance_sd"]:
    df[c] = df[c].map(lambda x: float(f"{x:.4g}"))

mlca_runtime.save_table(df, "interpretation.csv", caption="Validation permutation importance")
top = str(df.loc[0, "feature"])
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print(f"features={len(use_cols)} repeats=20 top={top}")
