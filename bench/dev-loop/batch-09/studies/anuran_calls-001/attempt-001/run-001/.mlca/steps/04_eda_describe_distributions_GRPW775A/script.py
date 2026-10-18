import json
import numpy as np
import pandas as pd
import mlca_runtime as rt

rt.set_seed(42)
inputs = rt.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group_col = proto.get("group_column")

df = pd.read_parquet(inputs["anuran_calls_uci"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = []
for c in df.columns:
    if c in drop:
        continue
    s = df[c]
    if pd.api.types.is_object_dtype(s) and len(s) > 0 and s.notna().sum() == len(s) and s.nunique() == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and len(s) > 0 and s.nunique() == len(s):
        continue
    feature_cols.append(c)

rows = []
n_num = n_cat = 0
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        n_num += 1
        rows.append({"group": "all", "column": c, "kind": "numeric",
                     "n": int(s.notna().sum()), "mean": float(s.mean()),
                     "sd": float(s.std()), "min": float(s.min()),
                     "median": float(s.median()), "max": float(s.max()),
                     "n_missing": int(s.isna().sum())})
    else:
        n_cat += 1
        vc = s.value_counts(dropna=True)
        tot = int(vc.sum())
        rows.append({"group": "all", "column": c, "kind": "categorical",
                     "n": tot, "n_levels": int(vc.shape[0]),
                     "top_share": float(vc.iloc[0] / tot) if tot else np.nan,
                     "n_missing": int(s.isna().sum())})

for t in targets:
    vc = df[t].value_counts(dropna=True)
    tot = int(vc.sum())
    for cls, n in vc.items():
        rows.append({"group": str(cls), "column": t, "kind": "target_class",
                     "n": int(n), "top_share": float(n / tot) if tot else np.nan})

cols = ["group", "column", "kind", "n", "mean", "sd", "min", "median",
        "max", "n_missing", "n_levels", "top_share"]
out = pd.DataFrame(rows)
for col in cols:
    if col not in out.columns:
        out[col] = np.nan
out = out[cols].head(50)
for col in ["n", "n_missing", "n_levels"]:
    out[col] = pd.to_numeric(out[col], errors="coerce")
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[col] = pd.to_numeric(out[col], errors="coerce").round(4)

rt.save_table(out, "distributions.csv")
rt.record_fact_candidate("n_numeric", int(n_num), source="distributions.csv")
rt.record_fact_candidate("n_categorical", int(n_cat), source="distributions.csv")
print(f"rows={len(out)} numeric={n_num} categorical={n_cat}")
