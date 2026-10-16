import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

feature_cols = []
for c in df.columns:
    if c in drop:
        continue
    s = df[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        if s.notna().sum() > 0 and s.nunique(dropna=True) == s.notna().sum():
            continue
    elif pd.api.types.is_integer_dtype(s):
        if len(s) > 0 and s.nunique(dropna=True) == len(s):
            continue
    feature_cols.append(c)

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        stripped = s.astype(str).str.strip()
        if stripped.isin(placeholders).any():
            ph_cols.append(c)
print(f"placeholder tokens found in {len(ph_cols)} columns: {ph_cols[:5]}")

rows = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        rows.append({
            "column": c, "type": "numeric",
            "n_missing": int(s.isna().sum()),
            "mean": s.mean(), "sd": s.std(),
            "min": s.min(), "median": s.median(), "max": s.max(),
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        top_share = float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan
        rows.append({
            "column": c, "type": "categorical",
            "n_missing": int(s.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": int(s.nunique(dropna=True)),
            "top_share": top_share,
        })

summary = pd.DataFrame(rows).head(50)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].astype(float).round(4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Column distributions")

n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
