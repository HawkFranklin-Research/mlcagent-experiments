import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys: {missing}. Available: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]
task = protocol["task"]

df = pd.read_parquet(inputs["adult_census_uci"])

drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)
feature_cols = [c for c in df.columns if c not in drop_cols]

def is_key_like(series, name):
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return series.nunique(dropna=False) == len(series)
    if pd.api.types.is_integer_dtype(series):
        if series.nunique(dropna=False) == len(series):
            lname = name.lower()
            return any(tok in lname for tok in ['id', 'code', 'key'])
    return False

key_like = [c for c in feature_cols if is_key_like(df[c], c)]
feature_cols = [c for c in feature_cols if c not in key_like]

placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
for c in feature_cols:
    if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
        s = df[c].astype(str).str.strip()
        if s.isin(placeholders).any():
            placeholder_cols.append(c)
print(f"Placeholder tokens found in {len(placeholder_cols)} columns: {placeholder_cols[:5]}")

def summarize(series, col, group_label):
    n_missing = series.isna().sum()
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return {
            "column": col,
            "type": "numeric",
            "group": group_label,
            "n_missing": n_missing,
            "mean": series.mean(),
            "sd": series.std(),
            "min": series.min(),
            "median": series.median(),
            "max": series.max(),
            "n_levels": np.nan,
            "top_share": np.nan,
        }
    else:
        vc = series.value_counts(dropna=True)
        n_levels = series.nunique(dropna=True)
        top_share = vc.iloc[0] / len(series) if len(vc) > 0 else np.nan
        return {
            "column": col,
            "type": "categorical",
            "group": group_label,
            "n_missing": n_missing,
            "mean": np.nan,
            "sd": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "n_levels": n_levels,
            "top_share": top_share,
        }

rows = []
for c in df.columns:
    rows.append(summarize(df[c], c, "overall"))

if len(targets) == 1:
    target = targets[0]
    for cls in df[target].dropna().unique():
        sub = df[df[target] == cls]
        for c in feature_cols:
            rows.append(summarize(sub[c], c, f"{target}={cls}"))

summary = pd.DataFrame(rows)
if len(summary) > 50:
    summary = summary.head(50)

for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    if c in summary.columns:
        summary[c] = summary[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)

mlca_runtime.save_table(summary, "distributions.csv", caption="Distribution summaries")

n_numeric = sum(1 for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]))
n_categorical = sum(1 for c in feature_cols if not (pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])))
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Rows: {len(df)}, Cols: {len(df.columns)}")
print(f"Numeric features: {n_numeric}, Categorical features: {n_categorical}")
print(f"Saved distributions.csv with {len(summary)} rows")
