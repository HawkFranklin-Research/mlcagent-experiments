import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol or "excluded_columns" not in protocol:
    raise KeyError(f"Protocol missing required keys; available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

df = pd.read_parquet(inputs["abalone_uci"])
features = [c for c in df.columns if c not in drop]

tokens = {"?", "-", "NA", "N/A", "na", "n/a"}
placeholder_cols = 0
rows = []

for col in features:
    ser = df[col]
    is_num = pd.api.types.is_numeric_dtype(ser) and not pd.api.types.is_bool_dtype(ser)
    if is_num:
        rows.append({
            "column": col, "kind": "numeric", "n": int(ser.notna().sum()),
            "n_missing": int(ser.isna().sum()), "mean": ser.mean(), "sd": ser.std(),
            "min": ser.min(), "median": ser.median(), "max": ser.max(),
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        s = ser.astype("string").str.strip()
        is_ph = (s.isin(tokens) | s.eq("")).fillna(False)
        if is_ph.any():
            placeholder_cols += 1
        clean = s.mask(is_ph)
        vc = clean.value_counts(normalize=True)
        rows.append({
            "column": col, "kind": "categorical", "n": int(clean.notna().sum()),
            "n_missing": int(clean.isna().sum()), "mean": np.nan, "sd": np.nan,
            "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": int(clean.nunique()), "top_share": float(vc.iloc[0]) if len(vc) else np.nan,
        })

dist = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[c] = dist[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)

n_numeric = int((dist["kind"] == "numeric").sum())
n_categorical = int((dist["kind"] == "categorical").sum())

mlca_runtime.save_table(dist, "distributions.csv", caption="Distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"features={len(features)} numeric={n_numeric} categorical={n_categorical}")
print(f"placeholder columns={placeholder_cols}")
print(f"summary rows={len(dist)}")
