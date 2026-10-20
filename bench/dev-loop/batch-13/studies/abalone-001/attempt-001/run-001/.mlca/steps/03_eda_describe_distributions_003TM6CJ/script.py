import json
import pandas as pd
import numpy as np
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing '{key}'. Available top-level keys: {list(d)}")
    return d[key]

targets = need(proto, "targets")
excluded = need(proto, "excluded_columns")
group = proto.get("group_column", None)

df = pd.read_parquet(paths["abalone_uci"])

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]

# Drop key-like columns: all-distinct integers/text codes with no measurement
drop_keys = []
for c in feature_cols:
    s = df[c].dropna()
    if len(s) == 0:
        continue
    if s.nunique() == len(s) and len(s) > 0:
        if s.dtype.kind in "iu" or s.dtype == object:
            drop_keys.append(c)
feature_cols = [c for c in feature_cols if c not in drop_keys]

# RL-0023: placeholder check on text columns
placeholders = {"?", "-", "NA", "N/A", ""}
ph_hits = 0
for c in feature_cols:
    if df[c].dtype == object:
        vals = df[c].astype(str).str.strip()
        mask = vals.isin(placeholders)
        if mask.any():
            ph_hits += 1
print(f"placeholder columns: {ph_hits}; dropped key-like: {len(drop_keys)}")

# Classify types
numeric_cols, cat_cols = [], []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
        numeric_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(df[c]):
        pass
    else:
        cat_cols.append(c)

sub = df[feature_cols]
rows = []
for c in numeric_cols:
    s = pd.to_numeric(sub[c], errors="coerce")
    rows.append({
        "variable": c, "type": "numeric",
        "n_missing": int(s.isna().sum()),
        "mean": s.mean(), "sd": s.std(), "min": s.min(),
        "median": s.median(), "max": s.max(),
        "n_levels": np.nan, "top_share": np.nan,
    })
for c in cat_cols:
    s = sub[c]
    vc = s.value_counts(dropna=True)
    n_levels = int(vc.shape[0])
    top_share = float(vc.iloc[0] / vc.sum()) if n_levels > 0 else np.nan
    rows.append({
        "variable": c, "type": "categorical",
        "n_missing": int(s.isna().sum()),
        "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan,
        "n_levels": n_levels, "top_share": top_share,
    })

summary = pd.DataFrame(rows, columns=["variable", "type", "n_missing", "mean", "sd",
                                       "min", "median", "max", "n_levels", "top_share"])
num_cols_r = {"mean", "sd", "min", "median", "max", "top_share"}
for c in num_cols_r:
    summary[c] = summary[c].astype(float).round(4)
summary = summary.head(50)

mlca_runtime.save_table(summary, "distributions.csv", caption="Per-variable distributions")

mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"numeric={len(numeric_cols)} categorical={len(cat_cols)}")
