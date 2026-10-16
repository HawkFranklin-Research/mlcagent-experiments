import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

df = pd.read_parquet(inputs["optical_digits_uci"])

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in df.columns:
    if df[c].dtype.kind in "OU":
        s = df[c].astype(str).str.strip()
        if s.isin(placeholders).any():
            ph_cols.append(c)
print(f"placeholder tokens in {len(ph_cols)} text columns")

def is_key_like(s):
    if s.dtype.kind in "OU":
        return s.nunique(dropna=True) == len(s)
    if s.dtype.kind in "iu":
        return s.nunique(dropna=True) == len(s)
    return False

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

features = [c for c in df.columns if c not in drop and not is_key_like(df[c])]

rows = []
n_numeric = 0
n_categorical = 0
for c in features:
    s = df[c]
    n_missing = int(s.isna().sum())
    if s.dtype.kind in "iuf":
        n_numeric += 1
        rows.append({
            "column": c, "kind": "numeric", "n_missing": n_missing,
            "mean": float(s.mean()), "sd": float(s.std()),
            "min": float(s.min()), "median": float(s.median()),
            "max": float(s.max()),
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        n_categorical += 1
        vc = s.value_counts(dropna=True)
        n_levels = int(vc.size)
        top_share = float(vc.iloc[0] / vc.sum()) if n_levels else np.nan
        rows.append({
            "column": c, "kind": "categorical", "n_missing": n_missing,
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": n_levels, "top_share": top_share,
        })

summary = pd.DataFrame(rows).round(4)
if len(summary) > 50:
    summary = summary.iloc[:50]
mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summary")

mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"features: {len(features)}")
print(f"numeric: {n_numeric}, categorical: {n_categorical}")
print(f"summary rows: {len(summary)}")
