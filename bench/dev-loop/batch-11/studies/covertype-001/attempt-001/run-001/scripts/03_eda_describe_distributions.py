import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["covertype_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available keys: {list(protocol.keys())}")
group_col = protocol.get("group_column")

all_cols = list(df.columns)
feature_cols = [
    c for c in all_cols
    if c not in targets and c not in excluded and (group_col is None or c != group_col)
]

nunique = {c: df[c].nunique(dropna=True) for c in feature_cols}
key_like = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        if nunique[c] == len(df) and len(df) > 0:
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s.dtype):
        if nunique[c] == len(df) and len(df) > 0:
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

placeholder_tokens = {"?", "-", "NA", "N/A"}
found_tokens = set()
placeholder_cols = 0
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        stripped = s.astype("string").str.strip()
        mask = stripped.isin(placeholder_tokens) | stripped.eq("")
        if mask.any():
            placeholder_cols += 1
            found_tokens.update(set(stripped[mask].dropna().unique()))
print(f"Placeholders {sorted(found_tokens)} in {placeholder_cols} text columns")

numeric_cols, cat_cols = [], []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        numeric_cols.append(c)
    elif pd.api.types.is_bool_dtype(s.dtype):
        cat_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s.dtype):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

rows = []
for c in numeric_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        s = (s - s.min()).dt.total_seconds() / 86400.0
    else:
        s = pd.to_numeric(s, errors="coerce")
    rows.append({
        "column": c, "type": "numeric",
        "n_missing": int(s.isna().sum()),
        "mean": s.mean(), "sd": s.std(), "min": s.min(),
        "median": s.median(), "max": s.max(),
        "n_levels": np.nan, "top_share": np.nan,
    })

for c in cat_cols:
    s = df[c].astype("string").str.strip()
    s = s.mask(s.isin(placeholder_tokens) | s.eq(""), pd.NA)
    vc = s.value_counts(dropna=True)
    denom = int(s.notna().sum())
    rows.append({
        "column": c, "type": "categorical",
        "n_missing": int(s.isna().sum()),
        "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan,
        "n_levels": int(vc.size),
        "top_share": (vc.iloc[0] / denom) if len(vc) > 0 and denom > 0 else np.nan,
    })

if len(rows) > 50:
    def priority(r):
        c = r["column"]
        s = df[c]
        is_bin = pd.api.types.is_numeric_dtype(s.dtype) and nunique.get(c, 0) <= 2
        if r["type"] == "numeric" and not is_bin:
            return 0
        if r["type"] == "categorical":
            return 1
        return 2
    rows = sorted(rows, key=priority)[:50]

dist = pd.DataFrame(rows)
num_header = ["mean", "sd", "min", "median", "max", "top_share"]
for c in num_header:
    dist[c] = dist[c].map(lambda x: x if pd.isna(x) else float(f"{x:.4g}"))

mlca_runtime.save_table(dist, "distributions.csv", caption="Feature distribution summaries")
n_numeric = int((dist["type"] == "numeric").sum())
n_categorical = int((dist["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Saved distributions.csv rows={len(dist)}")
print(f"n_numeric={n_numeric} n_categorical={n_categorical}")
