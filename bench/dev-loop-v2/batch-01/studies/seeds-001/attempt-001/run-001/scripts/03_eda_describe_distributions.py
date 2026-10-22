import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "task"]
missing = [k for k in required if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available keys: {list(proto)}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")
df = pd.read_parquet(inputs["seeds_uci"])

excluded_set = set(targets) | set(excluded)
if group:
    excluded_set.add(group)

feature_cols = []
for c in df.columns:
    if c in excluded_set:
        continue
    s = df[c]
    lname = str(c).lower()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s.dtype):
        non = s.dropna()
        if len(non) == len(s) and non.nunique() == len(s):
            continue
        if any(k in lname for k in ["id", "code", "key", "uuid"]):
            continue
    elif pd.api.types.is_integer_dtype(s.dtype):
        if s.nunique(dropna=True) == s.notna().sum() and any(k in lname for k in ["id", "code", "key"]):
            continue
    feature_cols.append(c)

numeric_cols = []
cat_cols = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

placeholders = {"?", "-", "NA", "N/A"}
ph_cols = 0
for c in cat_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s.dtype):
        v = s.dropna().astype(str).str.strip()
        if v.isin(placeholders).any() or (v == "").any():
            ph_cols += 1

def r4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    try:
        return float(f"{float(x):.4g}")
    except Exception:
        return x

rows = []
for c in numeric_cols:
    s = df[c]
    rows.append({
        "variable": c,
        "type": "numeric",
        "mean": r4(s.mean()),
        "sd": r4(s.std()),
        "min": r4(s.min()),
        "median": r4(s.median()),
        "max": r4(s.max()),
        "n_missing": int(s.isna().sum()),
        "n_levels": np.nan,
        "top_share": np.nan,
    })

for c in cat_cols:
    s = df[c]
    vc = s.value_counts(dropna=True, normalize=True)
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
        "top_share": r4(vc.iloc[0]) if len(vc) else np.nan,
    })

summary = pd.DataFrame(rows, columns=[
    "variable", "type", "mean", "sd", "min", "median",
    "max", "n_missing", "n_levels", "top_share"
])
mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"rows={len(df)}, features={len(feature_cols)}")
print(f"numeric={len(numeric_cols)}, categorical={len(cat_cols)}")
print(f"saved distributions.csv rows={len(summary)} placeholder_cols={ph_cols}")
