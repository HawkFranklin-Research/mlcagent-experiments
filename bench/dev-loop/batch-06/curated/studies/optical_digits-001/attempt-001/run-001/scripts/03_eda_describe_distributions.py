import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

df = pd.read_parquet(inputs["optical_digits_uci"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feature_cols = [c for c in df.columns if c not in drop]


def is_keylike(s):
    n = s.notna().sum()
    if n == 0:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == n
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == n
    return False


feature_cols = [c for c in feature_cols if not is_keylike(df[c])]

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        if stripped.isin(placeholders).any():
            ph_cols.append(c)
print(f"placeholders {sorted(placeholders)} found in {len(ph_cols)} columns")

rows = []
for c in feature_cols:
    s = df[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s):
        rows.append({
            "column": c, "type": "numeric",
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_missing": n_missing,
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        n_levels = int(vc.size)
        top_share = float(vc.iloc[0] / vc.sum()) if n_levels else np.nan
        rows.append({
            "column": c, "type": "categorical",
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_missing": n_missing,
            "n_levels": n_levels, "top_share": top_share,
        })

summary = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = pd.to_numeric(summary[c], errors="coerce").round(4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distributions")

n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"features: {len(feature_cols)} (numeric={n_numeric}, categorical={n_categorical})")
print(f"rows: {len(df)}")
