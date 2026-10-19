import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]

df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])

protected = set(targets) | set(excluded)
if group:
    protected.add(group)
feature_cols = [c for c in df.columns if c not in protected]

def is_date_like(s):
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return True
    if s.dtype == object:
        vals = s.dropna().astype(str)
        return len(vals) > 0 and pd.to_datetime(vals, errors="coerce").notna().all()
    return False

key_like = []
for c in feature_cols:
    s = df[c]
    if is_date_like(s):
        continue
    if s.nunique(dropna=True) == s.notna().sum() and not pd.api.types.is_float_dtype(s.dtype):
        key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

numeric_cols, categorical_cols, datetime_cols = [], [], []
for c in feature_cols:
    s = df[c]
    if is_date_like(s):
        datetime_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s.dtype):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

rows = []
for c in numeric_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append({
        "variable": c, "kind": "numeric", "n": len(s),
        "n_missing": int(s.isna().sum()),
        "mean": s.mean(), "sd": s.std(), "min": s.min(),
        "median": s.median(), "max": s.max(),
        "n_levels": np.nan, "top_share": np.nan,
    })
for c in categorical_cols:
    s = df[c]
    vc = s.dropna().value_counts(normalize=True)
    rows.append({
        "variable": c, "kind": "categorical", "n": len(s),
        "n_missing": int(s.isna().sum()),
        "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan,
        "n_levels": int(s.nunique(dropna=True)),
        "top_share": float(vc.iloc[0]) if len(vc) else np.nan,
    })

summary = pd.DataFrame(rows)
if len(summary) > 50:
    summary = summary.head(50)

def sig4(x):
    if pd.isna(x):
        return x
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    if col in summary.columns:
        summary[col] = summary[col].map(sig4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Numeric and categorical feature distributions")

n_numeric = int((summary["kind"] == "numeric").sum())
n_categorical = int((summary["kind"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

tokens = {"?", "-", "NA", "N/A", ""}
placeholder_cols = 0
for c in feature_cols:
    s = df[c]
    if s.dtype == object:
        vals = s.dropna().astype(str)
        bad = vals.isin(tokens) | vals.str.strip().eq("")
        if bad.any():
            placeholder_cols += 1

print(f"rows={len(df)} features={len(feature_cols)}")
print(f"numeric={n_numeric} categorical={n_categorical} date_skipped={len(datetime_cols)}")
print(f"text_columns_with_placeholders={placeholder_cols}")
