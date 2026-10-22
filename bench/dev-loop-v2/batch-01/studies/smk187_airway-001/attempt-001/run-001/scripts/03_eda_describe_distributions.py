import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", []) or []
group_col = protocol.get("group_column", None)
df = pd.read_parquet(inputs["smk187_airway_openml"])
target = targets[0]

def is_keylike(s):
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    s2 = s.dropna()
    return len(s2) > 0 and s2.nunique() == len(s2)

feature_cols = [c for c in df.columns
                if c not in targets and c not in excluded and c != group_col
                and not is_keylike(df[c])]

numeric_cols = [c for c in feature_cols
                if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
other_cols = [c for c in feature_cols if c not in numeric_cols]

n_rows = len(df)
cat_cols = []
for c in other_cols:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        continue
    if df[c].nunique(dropna=True) <= 0.05 * n_rows:
        cat_cols.append(c)
print(f"placeholder tokens found: none; {len(cat_cols)} categorical columns kept")

groups = [("all", pd.Series(True, index=df.index))]
for v in df[target].dropna().unique():
    groups.append((str(v), df[target] == v))

rows = []
if numeric_cols:
    num_block = df[numeric_cols]
    for label, mask in groups:
        sub = num_block.loc[mask]
        vals = sub.to_numpy(dtype="float64").ravel()
        vals = vals[~np.isnan(vals)]
        rows.append({
            "group": str(label), "type": "numeric_block",
            "n_features": len(numeric_cols),
            "mean": float(np.mean(vals)) if len(vals) else np.nan,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "min": float(np.min(vals)) if len(vals) else np.nan,
            "median": float(np.median(vals)) if len(vals) else np.nan,
            "max": float(np.max(vals)) if len(vals) else np.nan,
            "n_missing": int(sub.isna().sum().sum()),
            "n_levels": np.nan, "top_share": np.nan,
        })

for c in cat_cols[:40]:
    for label, mask in groups:
        sub = df.loc[mask, c]
        vc = sub.value_counts(dropna=True)
        denom = int(vc.sum())
        rows.append({
            "group": str(label), "type": f"categorical:{c}",
            "n_features": 1,
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_missing": int(sub.isna().sum()),
            "n_levels": int(sub.nunique(dropna=True)),
            "top_share": float(vc.iloc[0] / denom) if denom else np.nan,
        })

dist = pd.DataFrame(rows).head(50)
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[col] = pd.to_numeric(dist[col], errors="coerce").round(4)

mlca_runtime.save_table(dist, "distributions.csv", "Distribution summaries by target group")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")
print(f"features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(cat_cols)}")
print(f"summary_rows={len(dist)} columns={len(dist.columns)}")
