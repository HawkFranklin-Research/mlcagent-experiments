import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["wholesale_customers_uci"])
protocol = json.loads(paths["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available keys: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
task = protocol["task"]
group_col = protocol.get("group_column")

def is_key_like(s, name):
    vals = s.dropna()
    if vals.empty:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        vals = vals.astype(str)
        return vals.nunique() == len(vals) and len(vals) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return vals.nunique() == len(vals) and any(
            t in name.lower() for t in ["id", "key", "code", "index", "number"]
        )
    return False

feature_cols = [
    c for c in df.columns
    if c not in set(targets) | set(excluded) and c != group_col
]
feature_cols = [c for c in feature_cols if not is_key_like(df[c], c)]

text_cols = [
    c for c in feature_cols
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
]
placeholder_cols = []
for c in text_cols:
    s = df[c].astype("string").str.strip()
    mask = s.isin(["?", "-", "NA", "N/A", ""])
    if mask.any():
        placeholder_cols.append(c)
    df[c] = s.mask(mask, np.nan)
print(f"Placeholders found in {len(placeholder_cols)} text columns")

rows = []
for c in feature_cols:
    s = df[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        rows.append({
            "feature": c, "type": "numeric", "n_missing": n_missing,
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.dropna().value_counts(normalize=True)
        rows.append({
            "feature": c, "type": "categorical", "n_missing": n_missing,
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": int(s.nunique(dropna=True)),
            "top_share": float(vc.iloc[0]) if len(vc) else np.nan,
        })

summary = pd.DataFrame(rows).head(50).round(4)
mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summary")

n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Rows: {len(df)}, features: {len(feature_cols)}")
print(f"Numeric: {n_numeric}, categorical: {n_categorical}")
print(f"Saved distributions.csv rows={len(summary)}")
