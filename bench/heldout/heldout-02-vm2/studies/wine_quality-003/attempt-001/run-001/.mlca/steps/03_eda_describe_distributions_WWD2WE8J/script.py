import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["wine_quality_uci"])

try:
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
except KeyError as exc:
    raise KeyError(f"Missing protocol key. Available keys: {list(protocol)}") from exc
group_col = protocol.get("group_column")

def key_like(s):
    if s.isna().any() or s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s))

feature_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if key_like(df[c]):
        continue
    feature_cols.append(c)

numeric_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

def rs(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, np.integer)):
        return int(x)
    return float(f"{x:.4g}")

rows = []
for c in numeric_cols:
    s = df[c]
    rows.append({
        "variable": c, "kind": "numeric", "group": "overall",
        "n": int(s.notna().sum()), "mean": rs(s.mean()), "sd": rs(s.std()),
        "min": rs(s.min()), "median": rs(s.median()), "max": rs(s.max()),
        "n_missing": int(s.isna().sum()), "n_levels": np.nan, "top_share": np.nan,
    })
for c in categorical_cols:
    s = df[c]
    vc = s.value_counts(normalize=True, dropna=True)
    rows.append({
        "variable": c, "kind": "categorical", "group": "overall",
        "n": int(s.notna().sum()), "mean": np.nan, "sd": np.nan,
        "min": np.nan, "median": np.nan, "max": np.nan,
        "n_missing": int(s.isna().sum()), "n_levels": int(s.nunique(dropna=True)),
        "top_share": rs(vc.iloc[0] if len(vc) else np.nan),
    })

if targets and targets[0] in df.columns:
    t = targets[0]
    s = df[t]
    vc = s.value_counts(normalize=True, dropna=True)
    rows.append({
        "variable": t, "kind": "target", "group": "overall",
        "n": int(s.notna().sum()), "mean": np.nan, "sd": np.nan,
        "min": np.nan, "median": np.nan, "max": np.nan,
        "n_missing": int(s.isna().sum()), "n_levels": int(s.nunique(dropna=True)),
        "top_share": rs(vc.iloc[0] if len(vc) else np.nan),
    })
    if s.nunique(dropna=True) <= 20:
        counts = s.value_counts(dropna=True).sort_index()
        total = int(s.notna().sum())
        for level, cnt in counts.items():
            rows.append({
                "variable": t, "kind": "target_class", "group": str(level),
                "n": int(cnt), "mean": np.nan, "sd": np.nan,
                "min": np.nan, "median": np.nan, "max": np.nan,
                "n_missing": np.nan, "n_levels": np.nan,
                "top_share": rs(cnt / total if total else np.nan),
            })

cols = ["variable", "kind", "group", "n", "mean", "sd", "min", "median",
        "max", "n_missing", "n_levels", "top_share"]
dist = pd.DataFrame(rows, columns=cols).head(50)

mlca_runtime.save_table(dist, "distributions.csv", caption="Distribution summary by feature and target class")
n_num = len(numeric_cols)
n_cat = len(categorical_cols)
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")
print(f"rows={len(df)} features={len(feature_cols)} numeric={n_num} categorical={n_cat}")
print(f"saved distributions.csv rows={len(dist)}")
