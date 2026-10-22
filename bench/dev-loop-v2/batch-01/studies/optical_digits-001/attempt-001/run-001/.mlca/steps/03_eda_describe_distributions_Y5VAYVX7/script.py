import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
df = pd.read_parquet(inputs["optical_digits_uci"])

def key_like(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [
    c for c in df.columns
    if c not in targets and c not in excluded and c != group and not key_like(df[c])
]

numeric_cols = []
categorical_cols = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

rows = []
for c in numeric_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append({
        "feature": c, "type": "numeric",
        "mean": s.mean(), "sd": s.std(), "min": s.min(),
        "median": s.median(), "max": s.max(),
        "n_missing": int(s.isna().sum()),
        "n_levels": np.nan, "top_share": np.nan,
    })
for c in categorical_cols:
    s = df[c]
    vc = s.value_counts(normalize=True, dropna=True)
    rows.append({
        "feature": c, "type": "categorical",
        "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan,
        "n_missing": int(s.isna().sum()),
        "n_levels": int(s.nunique(dropna=True)),
        "top_share": float(vc.iloc[0]) if len(vc) else np.nan,
    })

dist = pd.DataFrame(rows)
if len(dist) > 50:
    head = dist.iloc[:49].copy()
    tail = dist.iloc[49:]
    agg = {
        "feature": f"other_features_{len(tail)}",
        "type": "mixed",
        "mean": tail["mean"].mean(),
        "sd": tail["sd"].mean(),
        "min": tail["min"].min(),
        "median": tail["median"].median(),
        "max": tail["max"].max(),
        "n_missing": int(tail["n_missing"].sum()),
        "n_levels": tail["n_levels"].sum(min_count=1),
        "top_share": tail["top_share"].mean(),
    }
    dist = pd.concat([head, pd.DataFrame([agg])], ignore_index=True)

for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[col] = dist[col].apply(
        lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float, np.floating)) else x
    )

mlca_runtime.save_table(dist, "distributions.csv", caption="Feature distribution summary")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

print(f"rows={len(df)} features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
print(f"distributions.csv rows={len(dist)} cols={dist.shape[1]}")
