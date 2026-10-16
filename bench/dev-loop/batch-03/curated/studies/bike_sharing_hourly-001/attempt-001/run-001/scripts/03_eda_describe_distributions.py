import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])

targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column")

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
for c in df.columns:
    if df[c].dtype == object:
        s = df[c].astype(str).str.strip()
        if s.isin(placeholders).any():
            ph_cols += 1
print(f"placeholder tokens found in {ph_cols} text columns")

feat_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]
keep = []
for c in feat_cols:
    s = df[c]
    if s.dtype == object and s.nunique(dropna=True) == len(s):
        continue
    keep.append(c)
feat_cols = keep

rows = []
for c in feat_cols:
    s = df[c]
    nm = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s):
        rows.append({"group": "overall", "variable": c, "type": "numeric",
                     "mean": s.mean(), "sd": s.std(), "min": s.min(),
                     "median": s.median(), "max": s.max(), "n_missing": nm,
                     "n_levels": np.nan, "top_share": np.nan})
    else:
        vc = s.value_counts(dropna=True)
        top = float(vc.iloc[0] / vc.sum()) if len(vc) else np.nan
        rows.append({"group": "overall", "variable": c, "type": "categorical",
                     "mean": np.nan, "sd": np.nan, "min": np.nan,
                     "median": np.nan, "max": np.nan, "n_missing": nm,
                     "n_levels": int(vc.size), "top_share": top})

out = pd.DataFrame(rows).head(50)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].astype(float).round(4)

mlca_runtime.save_table(out, "distributions.csv", caption="Column distributions")

n_num = int((out["type"] == "numeric").sum())
n_cat = int((out["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"numeric={n_num} categorical={n_cat} rows={len(out)}")
