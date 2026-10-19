import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["anuran_calls_uci"])

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing key {k!r}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns") or []
group_col = req(protocol, "group_column")
_ = req(protocol, "task")

if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded, str):
    excluded = [excluded]
missing_targets = [c for c in targets if c not in df.columns]
if missing_targets:
    raise KeyError(f"Target columns not found: {missing_targets}")

protected = set(targets) | set(excluded)
if group_col:
    protected.add(group_col)

def is_key_like(s: pd.Series) -> bool:
    n = len(s)
    if n == 0 or s.isna().any() or s.nunique(dropna=True) != n:
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    return (pd.api.types.is_integer_dtype(s) or
            pd.api.types.is_object_dtype(s) or
            pd.api.types.is_string_dtype(s))

feature_cols = []
for c in df.columns:
    if c in protected:
        continue
    if is_key_like(df[c]):
        continue
    feature_cols.append(c)

placeholders = {"?", "-", "NA", "N/A", "na", "n/a"}
placeholder_cols = 0
for c in df.columns:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str).str.strip()
        if vals.isin(placeholders | {""}).any():
            placeholder_cols += 1

def r4(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, np.integer)):
        return int(v)
    return float(f"{float(v):.4g}")

def cat_summary(s):
    sv = s.dropna()
    n = int(len(sv))
    n_missing = int(s.isna().sum())
    n_levels = int(sv.nunique()) if n else 0
    top_share = float(sv.value_counts().iloc[0] / n) if n else np.nan
    return n, n_missing, n_levels, top_share

rows = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        typ = "datetime"
        rows.append({"variable": c, "role": "feature", "type": typ,
                     "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
                     "mean": np.nan, "sd": np.nan, "min": np.nan,
                     "median": np.nan, "max": np.nan,
                     "n_levels": np.nan, "top_share": np.nan})
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        sv = pd.to_numeric(s, errors="coerce")
        rows.append({"variable": c, "role": "feature", "type": "numeric",
                     "n": int(sv.notna().sum()), "n_missing": int(sv.isna().sum()),
                     "mean": sv.mean(), "sd": sv.std(), "min": sv.min(),
                     "median": sv.median(), "max": sv.max(),
                     "n_levels": np.nan, "top_share": np.nan})
    else:
        n, n_missing, n_levels, top_share = cat_summary(s)
        rows.append({"variable": c, "role": "feature", "type": "categorical",
                     "n": n, "n_missing": n_missing,
                     "mean": np.nan, "sd": np.nan, "min": np.nan,
                     "median": np.nan, "max": np.nan,
                     "n_levels": n_levels, "top_share": top_share})

for c in targets:
    n, n_missing, n_levels, top_share = cat_summary(df[c])
    rows.append({"variable": c, "role": "target", "type": "categorical",
                 "n": n, "n_missing": n_missing,
                 "mean": np.nan, "sd": np.nan, "min": np.nan,
                 "median": np.nan, "max": np.nan,
                 "n_levels": n_levels, "top_share": top_share})

cols = ["variable", "role", "type", "n", "n_missing", "mean", "sd",
        "min", "median", "max", "n_levels", "top_share"]
summary = pd.DataFrame(rows, columns=cols)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].map(r4)
summary["n_levels"] = summary["n_levels"].map(lambda v: np.nan if pd.isna(v) else int(v))

if len(summary) > 50:
    tgt = summary[summary["role"] == "target"]
    feat = summary[summary["role"] == "feature"].head(max(0, 50 - len(tgt)))
    summary = pd.concat([feat, tgt], ignore_index=True)

mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Column distributions; target rows summarize outcome classes.")

feat_rows = summary[summary["role"] == "feature"]
n_numeric = int((feat_rows["type"] == "numeric").sum())
n_categorical = int((feat_rows["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)}, features={len(feature_cols)}, summary_rows={len(summary)}")
print(f"text placeholder columns={placeholder_cols}")
print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
