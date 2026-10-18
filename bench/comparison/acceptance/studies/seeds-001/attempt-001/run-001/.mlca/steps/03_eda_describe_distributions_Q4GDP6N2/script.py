import json
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["seeds_uci"])

if "targets" not in protocol or "excluded_columns" not in protocol:
    raise KeyError(f"Missing required protocol keys; available keys: {list(protocol.keys())}")

targets = list(protocol.get("targets") or [])
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
protected = set(targets) | set(excluded)
if group:
    protected.add(group)

def is_key_like(series, name):
    n = len(series)
    if n == 0:
        return False
    if is_object_dtype(series) or is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return series.dropna().nunique() == n and series.isna().sum() == 0
    lname = str(name).lower()
    if ("id" in lname or "code" in lname or "key" in lname) and series.dropna().nunique() == n:
        return True
    return False

feature_cols = [c for c in df.columns if c not in protected and not is_key_like(df[c], c)]
numeric_cols = [c for c in feature_cols if is_numeric_dtype(df[c]) and not isinstance(df[c].dtype, pd.CategoricalDtype)]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

placeholder_tokens = {"?", "-", "NA", "N/A", ""}
placeholder_cols = 0
for c in categorical_cols:
    s = df[c]
    if is_object_dtype(s) or is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        if stripped.isna().any() or stripped.isin(placeholder_tokens).any():
            placeholder_cols += 1

def sig(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(f"{float(x):.4g}")
    except Exception:
        return np.nan

rows = []
for c in numeric_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append({
        "variable": c,
        "type": "numeric",
        "mean": sig(s.mean()),
        "sd": sig(s.std()),
        "min": sig(s.min()),
        "median": sig(s.median()),
        "max": sig(s.max()),
        "n_missing": int(s.isna().sum()),
        "n_levels": np.nan,
        "top_level_share": np.nan,
    })
for c in categorical_cols:
    s = df[c]
    vc = s.value_counts(normalize=True, dropna=True)
    rows.append({
        "variable": c,
        "type": "categorical",
        "mean": np.nan,
        "sd": np.nan,
        "min": np.nan,
        "median": np.nan,
        "max": np.nan,
        "n_missing": int(s.isna().sum()),
        "n_levels": int(s.nunique(dropna=True)),
        "top_level_share": sig(vc.iloc[0]) if len(vc) else np.nan,
    })

summary = pd.DataFrame(rows, columns=[
    "variable", "type", "mean", "sd", "min", "median", "max",
    "n_missing", "n_levels", "top_level_share"
])
mlca_runtime.save_table(summary, "distributions.csv", caption="Column distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

print(f"rows={len(df)} features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
print(f"saved distributions.csv rows={len(summary)}")
if placeholder_cols:
    print(f"placeholder-like text found in {placeholder_cols} columns")
