import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())

required = ["targets", "excluded_columns", "task", "group_column"]
missing = [k for k in required if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(proto)}")

targets = proto["targets"]
excluded = set(proto["excluded_columns"])
group_col = proto["group_column"]

df = pd.read_parquet(paths["discovery"])
for t in targets:
    if t not in df.columns:
        raise KeyError(f"Target column {t!r} not found")

feat = [c for c in df.columns if c not in targets and c not in excluded]
if group_col is not None and group_col in df.columns:
    feat = [c for c in feat if c != group_col]

# Drop text-code identifiers: object/string columns whose values are all distinct
key_like = []
for c in feat:
    s = df[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        if s.nunique(dropna=True) == len(df):
            key_like.append(c)
feat = [c for c in feat if c not in key_like]

numeric_cols = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feat if c not in numeric_cols]

rows = [
    {"feature": "__n_numeric__", "type": "count", "scope": "all", "value": len(numeric_cols)},
    {"feature": "__n_categorical__", "type": "count", "scope": "all", "value": len(cat_cols)},
]

for c in numeric_cols[:15]:
    s = df[c]
    rows.append({
        "feature": c, "type": "numeric", "scope": "all",
        "n_missing": int(s.isna().sum()), "mean": s.mean(), "sd": s.std(),
        "min": s.min(), "median": s.median(), "max": s.max(),
        "n_levels": np.nan, "top_share": np.nan, "value": np.nan,
    })

for c in cat_cols[:10]:
    s = df[c]
    vc = s.value_counts(dropna=True)
    top_share = float(vc.iloc[0] / len(df)) if len(vc) else np.nan
    rows.append({
        "feature": c, "type": "categorical", "scope": "all",
        "n_missing": int(s.isna().sum()), "mean": np.nan, "sd": np.nan,
        "min": np.nan, "median": np.nan, "max": np.nan,
        "n_levels": int(s.nunique(dropna=True)), "top_share": top_share, "value": np.nan,
    })

for t in targets:
    vc = df[t].value_counts(dropna=True)
    n_levels = int(df[t].nunique(dropna=True))
    for cls, cnt in vc.items():
        rows.append({
            "feature": t, "type": "target_class", "scope": str(cls),
            "n_missing": int(df[t].isna().sum()), "mean": np.nan, "sd": np.nan,
            "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": n_levels, "top_share": float(cnt / len(df)), "value": np.nan,
        })

dist = pd.DataFrame(rows)[
    ["feature", "type", "scope", "n_missing", "mean", "sd", "min",
     "median", "max", "n_levels", "top_share", "value"]
]
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[c] = dist[c].astype(float).round(4)

if len(dist) > 50:
    dist = dist.head(50)

mlca_runtime.save_table(dist, "distributions.csv", caption="Column distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"rows={len(df)} features={len(feat)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
print(f"summary rows={len(dist)}")
