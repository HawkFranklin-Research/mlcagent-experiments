import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs["_protocol"].read_text())
try:
    targets = proto["targets"]
    excluded = proto["excluded_columns"]
    task = proto["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available: {list(proto.keys())}")
group = proto.get("group_column", None)

df = pd.read_parquet(inputs["seeds_uci"])
print(f"shape={df.shape}")

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feat_cols = [c for c in df.columns if c not in drop]

n = len(df)
key_like = []
for c in feat_cols:
    s = df[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s)):
        if n > 1 and s.nunique(dropna=True) == n:
            key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]

numeric = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
categorical = [c for c in feat_cols if c not in numeric]

rows = []
for c in numeric:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append({"column": c, "type": "numeric",
                 "mean": s.mean(), "sd": s.std(), "min": s.min(),
                 "median": s.median(), "max": s.max(),
                 "n_missing": int(s.isna().sum()),
                 "n_levels": np.nan, "top_share": np.nan})
for c in categorical:
    s = df[c]
    vc = s.value_counts(dropna=True)
    share = float(vc.iloc[0] / s.notna().sum()) if len(vc) else np.nan
    rows.append({"column": c, "type": "categorical",
                 "mean": np.nan, "sd": np.nan, "min": np.nan,
                 "median": np.nan, "max": np.nan,
                 "n_missing": int(s.isna().sum()),
                 "n_levels": int(s.nunique()), "top_share": share})

summary = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].apply(lambda v: float(f"{v:.4g}") if pd.notna(v) else v)
mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Per-feature distribution summary")

mlca_runtime.record_fact_candidate("n_numeric", len(numeric), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical), source="distributions.csv")
print(f"numeric={len(numeric)} categorical={len(categorical)}")
