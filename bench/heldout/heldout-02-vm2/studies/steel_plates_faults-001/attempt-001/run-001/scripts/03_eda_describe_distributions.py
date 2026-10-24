import json
import numpy as np
import pandas as pd
import mlca_runtime
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["steel_plates_faults_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol:
    raise KeyError(f"Missing top-level 'targets'; available keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]

def sig4(v):
    if pd.isna(v):
        return v
    try:
        return float(f"{float(v):.4g}")
    except Exception:
        return v

rows = []
n_num = n_cat = 0
for c in df.columns:
    s = df[c]
    if is_numeric_dtype(s):
        kind = "numeric"
        n_num += 1
        rows.append({
            "variable": c, "kind": kind, "group": "overall", "group_class": np.nan,
            "n_missing": int(s.isna().sum()), "mean": s.mean(), "sd": s.std(),
            "min": s.min(), "median": s.median(), "max": s.max(),
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        kind = "categorical"
        n_cat += 1
        vc = s.value_counts(normalize=True, dropna=True)
        rows.append({
            "variable": c, "kind": kind, "group": "overall", "group_class": np.nan,
            "n_missing": int(s.isna().sum()), "mean": np.nan, "sd": np.nan,
            "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": int(s.nunique(dropna=True)),
            "top_share": float(vc.iloc[0]) if len(vc) else np.nan,
        })

for t in targets:
    if t not in df.columns:
        continue
    s = df[t]
    vc = s.value_counts(normalize=True, dropna=True)
    for cls, share in vc.items():
        if len(rows) >= 50:
            break
        rows.append({
            "variable": t, "kind": "categorical", "group": t, "group_class": cls,
            "n_missing": int(s.isna().sum()), "mean": np.nan, "sd": np.nan,
            "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": 1, "top_share": float(share),
        })
    if len(rows) >= 50:
        break

out = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].map(sig4)

placeholders = {"?", "-", "NA", "N/A"}
bad_cols = 0
for c in df.columns:
    if is_object_dtype(df[c]) or is_string_dtype(df[c]):
        txt = df[c].dropna().astype(str).str.strip()
        if txt.isin(placeholders).any() or (txt == "").any():
            bad_cols += 1

print(f"shape {df.shape} numeric {n_num} categorical {n_cat}")
print(f"placeholder text columns {bad_cols}")
mlca_runtime.save_table(out, "distributions.csv", caption="Distribution summaries by variable and target class")
mlca_runtime.record_fact_candidate("n_numeric", int(n_num), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(n_cat), source="distributions.csv")
print(f"saved distributions.csv rows {len(out)}")
