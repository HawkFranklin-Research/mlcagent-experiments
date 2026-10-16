import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["anuran_calls_uci"])

targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column")

def is_keylike(s):
    if s.dtype.kind in "iu" and s.nunique() == len(s):
        return True
    if s.dtype == object and s.nunique() == len(s):
        return True
    return False

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col and not is_keylike(df[c])]

rows = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        rows.append({
            "variable": c, "kind": "numeric", "group": "overall",
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_missing": int(s.isna().sum()),
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        rows.append({
            "variable": c, "kind": "categorical", "group": "overall",
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_missing": int(s.isna().sum()),
            "n_levels": int(s.nunique()),
            "top_share": float(vc.iloc[0] / len(s)) if len(vc) else np.nan,
        })

n_numeric = sum(1 for r in rows if r["kind"] == "numeric")
n_categorical = sum(1 for r in rows if r["kind"] == "categorical")

for t in targets:
    vc = df[t].value_counts(dropna=True).head(5)
    for cls, cnt in vc.items():
        rows.append({
            "variable": t, "kind": "target_class", "group": str(cls),
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_missing": int(df[t].isna().sum()),
            "n_levels": int(df[t].nunique()),
            "top_share": float(cnt / len(df)),
        })

summary = pd.DataFrame(rows).head(50)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].astype(float).round(4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Column distributions")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"rows={len(df)} features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
