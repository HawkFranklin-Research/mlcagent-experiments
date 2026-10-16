import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["optical_digits_uci"])

targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
reserved = set(targets) | set(excluded)
if group:
    reserved.add(group)

feature_cols = []
for c in df.columns:
    if c in reserved:
        continue
    s = df[c]
    if pd.api.types.is_integer_dtype(s) and s.notna().all() and s.nunique() == len(s):
        continue
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == s.notna().sum() and s.notna().sum() > 0:
            continue
    feature_cols.append(c)

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in feature_cols:
    s = df[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str).str.strip()
        if vals.isin(placeholders).any():
            ph_cols.append(c)
print(f"placeholder tokens in {len(ph_cols)} columns")

rows = []
n_numeric = 0
n_categorical = 0
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        n_numeric += 1
        rows.append({"column": c, "type": "numeric",
                     "mean": s.mean(), "sd": s.std(), "min": s.min(),
                     "median": s.median(), "max": s.max(),
                     "n_missing": int(s.isna().sum()),
                     "n_levels": np.nan, "top_share": np.nan})
    else:
        n_categorical += 1
        vc = s.value_counts(dropna=True)
        n_levels = int(vc.shape[0])
        top_share = float(vc.iloc[0] / vc.sum()) if n_levels > 0 else np.nan
        rows.append({"column": c, "type": "categorical",
                     "mean": np.nan, "sd": np.nan, "min": np.nan,
                     "median": np.nan, "max": np.nan,
                     "n_missing": int(s.isna().sum()),
                     "n_levels": n_levels, "top_share": top_share})

summary = pd.DataFrame(rows)
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[col] = summary[col].astype(float).round(4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distributions")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
