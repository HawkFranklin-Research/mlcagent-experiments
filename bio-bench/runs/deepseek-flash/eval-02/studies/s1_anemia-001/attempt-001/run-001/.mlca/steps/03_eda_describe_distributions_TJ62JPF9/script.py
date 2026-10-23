import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol.get("targets", [])
if not targets:
    raise KeyError(f"Missing targets; available keys: {list(protocol.keys())}")
target = targets[0]
excluded = set(protocol.get("excluded_columns", []) or [])
group_col = protocol.get("group_column")

df = pd.read_parquet(inputs["anemia"])
df = df.copy()

variables = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]

tokens = {"?", "-", "NA", "N/A", "", "na", "n/a", "null", "None"}
ph_cols = 0
ph_found = set()
for c in variables:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        s = df[c].astype("string").str.strip()
        mask = s.isin(tokens) & df[c].notna()
        if mask.any():
            ph_cols += 1
            ph_found.update(s[mask].dropna().unique().tolist())
        df[c] = s.where(~mask, pd.NA)

def is_num(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

def summarize(s, group, var):
    row = {"variable": var, "group": group, "n": int(s.notna().sum()),
           "n_missing": int(s.isna().sum()),
           "mean": np.nan, "sd": np.nan, "min": np.nan,
           "median": np.nan, "max": np.nan,
           "n_levels": np.nan, "top_share": np.nan}
    if is_num(s):
        row["dtype"] = "numeric"
        if pd.api.types.is_datetime64_any_dtype(s):
            vals = s.dropna().astype("int64")
        else:
            vals = pd.to_numeric(s, errors="coerce").dropna()
        if len(vals):
            row.update(mean=vals.mean(), sd=vals.std(), min=vals.min(),
                       median=vals.median(), max=vals.max())
    else:
        row["dtype"] = "categorical"
        v = s.dropna()
        if len(v):
            row["n_levels"] = int(v.nunique())
            row["top_share"] = float(v.value_counts(normalize=True).iloc[0])
    return row

numeric_vars = [c for c in variables if is_num(df[c])]
cat_vars = [c for c in variables if c not in numeric_vars]
n_numeric, n_categorical = len(numeric_vars), len(cat_vars)

n_classes = df[target].nunique(dropna=True)
include_classes = (2 <= n_classes <= 5) and (len(variables) * (1 + n_classes) <= 50)

rows = []
for var in variables:
    rows.append(summarize(df[var], "overall", var))
if include_classes:
    for cls in sorted(df[target].dropna().unique()):
        mask = df[target] == cls
        for var in variables:
            rows.append(summarize(df.loc[mask, var], f"{target}={cls}", var))

summary = pd.DataFrame(rows)
summary = summary.head(50)
cols = ["variable", "group", "dtype", "n", "n_missing", "mean", "sd",
        "min", "median", "max", "n_levels", "top_share"]
summary = summary[cols]

def sig4(x):
    if pd.isna(x):
        return x
    try:
        return float(f"{x:.4g}")
    except Exception:
        return x

for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].map(sig4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Column distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)}, features={len(variables)}, numeric={n_numeric}, categorical={n_categorical}")
print(f"placeholders found in {ph_cols} columns: {sorted(ph_found)}")
print(f"per-class rows included: {include_classes}")
print(f"distributions rows: {len(summary)}")
