import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["energy_efficiency_uci"])

targets = list(proto.get("targets", []))
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column")

rows = []
for c in df.columns:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        rows.append({
            "column": c, "type": "numeric",
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_missing": int(s.isna().sum()),
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        rows.append({
            "column": c, "type": "categorical",
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_missing": int(s.isna().sum()),
            "n_levels": int(vc.size),
            "top_share": float(vc.iloc[0] / len(s)) if len(vc) else np.nan,
        })

dist = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[c] = dist[c].astype(float).round(4)

mlca_runtime.save_table(dist, "distributions.csv", caption="Column distributions")

n_numeric = int((dist["type"] == "numeric").sum())
n_categorical = int((dist["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)} cols={df.shape[1]}")
print(f"n_numeric={n_numeric} n_categorical={n_categorical}")
