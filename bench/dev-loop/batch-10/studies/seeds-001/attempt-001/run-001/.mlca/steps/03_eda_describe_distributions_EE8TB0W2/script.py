import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["seeds_uci"])

targets = list(protocol.get("targets", []))
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column", None)

# Determine feature columns (RL-0007): all except targets/excluded/group and key-like.
blocked = set(targets) | set(excluded)
if group is not None:
    blocked.add(group)
n = len(df)
feat_cols = []
for c in df.columns:
    if c in blocked:
        continue
    s = df[c]
    if s.dtype.kind in ("i", "u", "f"):
        # integer sequence with all distinct values and no measurement -> key-like
        if s.dtype.kind in ("i", "u") and s.nunique() == n and n > 1:
            continue
    feat_cols.append(c)

# Sort columns into numeric vs categorical (RL-0010)
num_cols, cat_cols = [], []
for c in feat_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        cat_cols.append(c)

rows = []
for c in num_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append({
        "column": c,
        "type": "numeric",
        "mean": s.mean(),
        "sd": s.std(),
        "min": s.min(),
        "median": s.median(),
        "max": s.max(),
        "n_missing": int(s.isna().sum()),
        "n_levels": np.nan,
        "top_level_share": np.nan,
    })
for c in cat_cols:
    s = df[c]
    nn = s.dropna()
    nl = int(nn.nunique())
    if len(nn) > 0:
        top_share = float(nn.value_counts(normalize=True, dropna=True).iloc[0])
    else:
        top_share = np.nan
    rows.append({
        "column": c,
        "type": "categorical",
        "mean": np.nan,
        "sd": np.nan,
        "min": np.nan,
        "median": np.nan,
        "max": np.nan,
        "n_missing": int(s.isna().sum()),
        "n_levels": nl,
        "top_level_share": top_share,
    })

summary = pd.DataFrame(rows)
if len(summary) > 50:
    summary = summary.head(50)
summary = summary[["column", "type", "mean", "sd", "min", "median", "max",
                   "n_missing", "n_levels", "top_level_share"]]
for c in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    summary[c] = summary[c].astype(float).round(4)

mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Numeric and categorical feature distributions")

mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print("features:", len(feat_cols), "numeric:", len(num_cols), "categorical:", len(cat_cols))
