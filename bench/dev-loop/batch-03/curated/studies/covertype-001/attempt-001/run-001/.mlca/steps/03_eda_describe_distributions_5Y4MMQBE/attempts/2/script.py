import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = list(proto["targets"])
excluded = list(proto.get("excluded_columns", []))

df = pd.read_parquet(inputs["covertype_uci"])

rows = []
for col in df.columns:
    s = df[col]
    is_num = pd.api.types.is_numeric_dtype(s) and col not in targets and s.nunique() > 2
    if is_num:
        rows.append({"column": col, "kind": "numeric",
                     "mean": s.mean(), "sd": s.std(), "min": s.min(),
                     "median": s.median(), "max": s.max(),
                     "n_missing": int(s.isna().sum()),
                     "n_levels": np.nan, "top_share": np.nan})
    else:
        vc = s.value_counts(dropna=True)
        rows.append({"column": col, "kind": "categorical",
                     "mean": np.nan, "sd": np.nan, "min": np.nan,
                     "median": np.nan, "max": np.nan,
                     "n_missing": int(s.isna().sum()),
                     "n_levels": int(vc.size),
                     "top_share": float(vc.iloc[0] / vc.sum()) if vc.size else np.nan})

out = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(out, "distributions.csv", caption="Column distributions")

n_num = int((out["kind"] == "numeric").sum())
n_cat = int((out["kind"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"numeric={n_num} categorical={n_cat} rows={len(out)}")
