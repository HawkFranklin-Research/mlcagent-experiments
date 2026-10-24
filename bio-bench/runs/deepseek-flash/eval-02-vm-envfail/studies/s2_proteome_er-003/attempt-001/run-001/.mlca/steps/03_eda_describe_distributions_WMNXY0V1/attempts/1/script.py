import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in proto]
if missing:
    raise ValueError(f"Missing protocol keys {missing}; available top-level keys: {sorted(proto)}")

targets = list(proto["targets"])
excluded = set(proto["excluded_columns"]) if isinstance(proto["excluded_columns"], list) else set()
if not isinstance(proto["excluded_columns"], list):
    raise ValueError("_protocol['excluded_columns'] must be a list")
group_col = proto["group_column"]
task = proto["task"]
target = targets[0]

df = pd.read_parquet(inp["proteome_er"])
if target not in df.columns:
    raise ValueError(f"Target column {target!r} not found in input data")

feature_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = df[c]
    if len(df) > 1 and s.nunique(dropna=True) == len(df):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s):
            continue
    feature_cols.append(c)

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

placeholders = {"?", "-", "NA", "N/A"}
ph_cols = 0
for c in cat_cols:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        s = df[c].astype("string").str.strip()
        mask = s.isin(placeholders) | (s == "")
        if mask.any():
            ph_cols += 1
print(f"placeholders found: ?, -, NA, N/A, empty in {ph_cols} text columns")

classes = sorted(df[target].dropna().unique().tolist())
if len(feature_cols) * (1 + len(classes)) <= 50:
    groups = [("overall", df)] + [(str(k), df[df[target] == k]) for k in classes]
    include_cols = feature_cols
else:
    groups = [("overall", df)]
    include_cols = feature_cols[:50]

rows = []
for gname, sub in groups:
    for c in include_cols:
        if len(rows) >= 50:
            break
        s = sub[c]
        n_missing = int(s.isna().sum())
        n = int(s.notna().sum())
        if c in numeric_cols:
            x = pd.to_numeric(s, errors="coerce")
            rows.append({
                "feature": c, "type": "numeric", "group": gname,
                "mean": x.mean(), "sd": x.std(ddof=1), "min": x.min(),
                "median": x.median(), "max": x.max(), "n_missing": n_missing,
                "n_levels": np.nan, "top_share": np.nan, "n": n,
            })
        else:
            vc = s.value_counts(dropna=True)
            top_share = (vc.iloc[0] / n) if n > 0 and len(vc) > 0 else np.nan
            rows.append({
                "feature": c, "type": "categorical", "group": gname,
                "mean": np.nan, "sd": np.nan, "min": np.nan,
                "median": np.nan, "max": np.nan, "n_missing": n_missing,
                "n_levels": int(vc.size), "top_share": top_share, "n": n,
            })
    if len(rows) >= 50:
        break

out = pd.DataFrame(rows)
stat_cols = ["mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share", "n"]
for c in stat_cols:
    out[c] = out[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
out["n_missing"] = out["n_missing"].astype("Int64")
out["n_levels"] = out["n_levels"].astype("Int64")
out["n"] = out["n"].astype("Int64")

mlca_runtime.save_table(out, "distributions.csv", caption="Feature distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
print(f"rows saved={len(out)}")
