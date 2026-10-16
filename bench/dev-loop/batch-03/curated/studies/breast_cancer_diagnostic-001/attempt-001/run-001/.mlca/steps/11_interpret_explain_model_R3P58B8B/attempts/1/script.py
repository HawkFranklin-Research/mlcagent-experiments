import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys {missing} in _protocol. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]
n_classes = task.get("n_classes", 2)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def is_key_like(series):
    if series.dtype == object or series.dtype == 'string':
        return series.nunique() == len(series)
    if pd.api.types.is_integer_dtype(series):
        return series.nunique() == len(series)
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not is_key_like(train[c])
]

X_val = val[feature_cols]
y_val = val[targets[0]]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != feature_cols:
        raise ValueError("Model feature names do not match computed feature columns.")

def scorer(estimator, X, y):
    proba = estimator.predict_proba(X)
    if n_classes == 2:
        return roc_auc_score(y, proba[:, 1], pos_label=estimator.classes_[1])
    return roc_auc_score(y, proba, multi_class="ovr", average="macro")

result = permutation_importance(
    model, X_val, y_val, scoring=scorer, n_repeats=20, random_state=42, n_jobs=1
)

df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
})
df = df.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
df = df.round(4)

mlca_runtime.save_table(df, "interpretation.csv", caption="Permutation importance top 15 features")
top_feature = df.iloc[0]["feature"]
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"Val shape: {X_val.shape}")
print(f"Top feature: {top_feature}")
