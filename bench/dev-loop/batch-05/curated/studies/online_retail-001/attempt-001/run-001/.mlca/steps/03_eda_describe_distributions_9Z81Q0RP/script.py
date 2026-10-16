import json
from math import floor, log10
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["online_retail_uci"])

if "group_column" not in protocol:
    raise KeyError(f"group_column missing. Available keys: {sorted(protocol.keys())}")
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol["group_column"]

drop = set(targets) | set(excluded) | {group}
candidates = [c for c in df.columns if c not in drop]

def is_keylike(s):
    if len(s) == 0:
        return False
    if s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    if pd.api.types.is_datetime64_any_dtype(s):
        return False
    if pd.api.types.is_bool_dtype(s):
        return False
    return True

features = [c for c in candidates if not is_keylike(df[c])]

placeholders = {"?", "-", "NA", "N/A", ""}
ph_found, ph_cols = set(), set()
for c in features:
    s = df[c]
    if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s)):
        stripped = s.astype(str).str.strip()
        for p in placeholders:
            if (stripped == p).any():
                ph_found.add(p if p else "<empty>")
                ph_cols.add(c)
print(f"placeholders {sorted(ph_found)} in {len(ph_cols)} columns")

def sig4(v):
    if pd.isna(v):
        return np.nan
    v = float(v)
    if v == 0:
        return 0.0
    return round(v, -int(floor(log10(abs(v)))) + 3)

n_numeric = sum(1 for c in features if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]))
n_categorical = len(features) - n_numeric

rows = []
for c in features:
    s = df[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append({
            "variable": c, "type": "numeric",
            "mean": sig4(s.mean()), "sd": sig4(s.std()),
            "min": sig4(s.min()), "median": sig4(s.median()),
            "max": sig4(s.max()), "n_missing": n_missing,
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        n_levels = int(vc.size)
        top_share = sig4(vc.iloc[0] / vc.sum()) if n_levels > 0 else np.nan
        rows.append({
            "variable": c, "type": "categorical",
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_missing": n_missing,
            "n_levels": n_levels, "top_share": top_share,
        })

summary = pd.DataFrame(rows)
if len(summary) > 50:
    summary = summary.head(50)

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"features: {len(features)}")
print(f"numeric: {n_numeric}, categorical: {n_categorical}")
print(f"summary shape: {summary.shape}")
