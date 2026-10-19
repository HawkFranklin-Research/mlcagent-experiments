import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns"]:
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column", None)

df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])

placeholder_tokens = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
for c in df.columns:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        s = df[c].dropna().astype(str).str.strip()
        if len(s) and s.isin(placeholder_tokens).any():
            placeholder_cols.append(c)
print(f"placeholder_columns={len(placeholder_cols)}")

excluded_set = set(targets) | set(excluded)
if group_col is not None:
    excluded_set.add(group_col)

feature_cols = [c for c in df.columns if c not in excluded_set]

kept = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        n_notna = s.notna().sum()
        n_unique = s.nunique(dropna=True)
        if n_notna > 0 and n_unique == n_notna and n_unique > 0.9 * len(df):
            continue
    kept.append(c)
feature_cols = kept

numeric_cols = []
categorical_cols = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        numeric_cols.append(c)
    elif (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_bool_dtype(s)
    ):
        categorical_cols.append(c)

rows = []
for c in numeric_cols:
    s = df[c]
    rows.append(
        {
            "variable": c,
            "type": "numeric",
            "group": "overall",
            "mean": s.mean(),
            "sd": s.std(),
            "min": s.min(),
            "median": s.median(),
            "max": s.max(),
            "n_missing": int(s.isna().sum()),
            "n_levels": np.nan,
            "top_share": np.nan,
        }
    )

for c in categorical_cols:
    s = df[c]
    vc = s.value_counts(dropna=True, normalize=True)
    rows.append(
        {
            "variable": c,
            "type": "categorical",
            "group": "overall",
            "mean": np.nan,
            "sd": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "n_missing": int(s.isna().sum()),
            "n_levels": int(s.nunique(dropna=True)),
            "top_share": float(vc.iloc[0]) if len(vc) else np.nan,
        }
    )

summary = pd.DataFrame(rows)

def sig4(x):
    if pd.isna(x) or x == 0:
        return x
    return float(f"{x:.4g}")

for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[col] = summary[col].map(sig4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Distribution summary of feature columns")

n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)} features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
