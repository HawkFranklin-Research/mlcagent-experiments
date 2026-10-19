import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns") or [])
group_col = protocol.get("group_column")

def keylike(s):
    if s.dtype == object:
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in df.columns if c not in drop and not keylike(df[c])]

def is_numeric(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

def r4(x):
    try:
        return float(f"{x:.4g}")
    except Exception:
        return x

binary_target = "heart_disease_binary" if "heart_disease_binary" in targets else targets[0]
groups = [("all", df)]
for v in sorted(df[binary_target].dropna().unique()):
    groups.append((f"{binary_target}={v}", df[df[binary_target] == v]))

rows = []
for col in feature_cols:
    numeric = is_numeric(df[col])
    for g, sub in groups:
        s = sub[col]
        rec = {
            "variable": col,
            "type": "numeric" if numeric else "categorical",
            "group": g,
            "n": int(len(s)),
            "n_missing": int(s.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": np.nan, "top_share": np.nan,
        }
        if numeric:
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v):
                rec["mean"] = r4(v.mean())
                rec["sd"] = r4(v.std()) if len(v) > 1 else np.nan
                rec["min"] = r4(v.min())
                rec["median"] = r4(v.median())
                rec["max"] = r4(v.max())
        else:
            vc = s.value_counts(dropna=True)
            rec["n_levels"] = int(vc.shape[0])
            if len(s) and vc.shape[0]:
                rec["top_share"] = r4(vc.iloc[0] / len(s))
        rows.append(rec)

out = pd.DataFrame(rows)
if len(out) > 50:
    out = out.head(50).reset_index(drop=True)
cols = ["variable", "type", "group", "n", "n_missing", "mean", "sd",
        "min", "median", "max", "n_levels", "top_share"]
out = out[cols]

mlca_runtime.save_table(out, "distributions.csv",
                        caption="Numeric and categorical summaries by target class")

base = out[out["group"] == "all"]
n_numeric = int((base["type"] == "numeric").sum())
n_categorical = int((base["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"rows={len(out)} numeric={n_numeric} categorical={n_categorical}")
