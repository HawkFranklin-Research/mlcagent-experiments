import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column", None)

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]

key_like = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        key_like.append(c)
    elif (pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s)) and s.nunique() == len(s):
        key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

rows = []
for c in feature_cols:
    s = df[c]
    n_missing = s.isna().sum()
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        row = {
            "variable": c, "type": "numeric",
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(), "n_missing": n_missing,
            "n_levels": np.nan, "top_level_share": np.nan,
        }
    else:
        n_levels = s.nunique(dropna=True)
        top_level_share = s.value_counts(normalize=True, dropna=True).iloc[0] if n_levels > 0 else np.nan
        row = {
            "variable": c, "type": "categorical",
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan, "n_missing": n_missing,
            "n_levels": n_levels, "top_level_share": top_level_share,
        }
    rows.append(row)

summary = pd.DataFrame(rows)
for col in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    summary[col] = summary[col].round(4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Distribution summaries")

n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Feature columns: {len(feature_cols)}")
print(f"Numeric: {n_numeric}, Categorical: {n_categorical}")
print(f"Key-like excluded: {len(key_like)}")
print(f"Summary shape: {summary.shape}")
