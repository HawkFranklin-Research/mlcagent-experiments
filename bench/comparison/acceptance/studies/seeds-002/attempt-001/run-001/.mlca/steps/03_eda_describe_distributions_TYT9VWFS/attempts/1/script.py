import json
import mlca_runtime
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def req(k):
    if k not in proto:
        raise KeyError(f"missing {k}; available: {sorted(proto.keys())}")
    return proto[k]

targets = req("targets")
excluded = req("excluded_columns")
group = proto.get("group_column")
_ = req("task")

df = pd.read_parquet(inp["seeds_uci"])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

features = []
for c in df.columns:
    if c in drop:
        continue
    s = df[c]
    if len(s) > 0 and s.nunique(dropna=False) == len(s) and (s.dtype == object or pd.api.types.is_integer_dtype(s)):
        continue
    features.append(c)

ph = {"?", "-", "NA", "N/A"}
ph_cols = sum(1 for c in features if df[c].dtype == object and df[c].astype(str).str.strip().isin(ph).any())
print(f"placeholder columns: {ph_cols}")

rows = []
n_num = n_cat = 0
for c in features:
    s = df[c]
    miss = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s):
        n_num += 1
        rows.append({"feature": c, "type": "numeric", "mean": s.mean(), "sd": s.std(),
                     "min": s.min(), "median": s.median(), "max": s.max(),
                     "n_levels": np.nan, "top_share": np.nan, "n_missing": miss})
    else:
        n_cat += 1
        vc = s.value_counts(dropna=False)
        rows.append({"feature": c, "type": "categorical", "mean": np.nan, "sd": np.nan,
                     "min": np.nan, "median": np.nan, "max": np.nan,
                     "n_levels": int(vc.shape[0]),
                     "top_share": float(vc.iloc[0] / len(s)) if len(s) else np.nan,
                     "n_missing": miss})

summary = pd.DataFrame(rows)
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[col] = summary[col].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)
summary["n_levels"] = summary["n_levels"].astype("Float64")

mlca_runtime.save_table(summary, "distributions.csv", caption="Per-feature distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", int(n_num), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(n_cat), source="distributions.csv")
print(f"features={len(features)} numeric={n_num} categorical={n_cat} rows={len(df)}")
