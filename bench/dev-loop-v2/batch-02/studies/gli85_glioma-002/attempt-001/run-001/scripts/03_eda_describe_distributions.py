import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["gli85_glioma_openml"])

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
blocked = set(targets) | set(excluded)
if group_col:
    blocked.add(group_col)

n_rows = len(df)
feature_cols = [c for c in df.columns if c not in blocked]

def key_like(c):
    s = df[c]
    if not s.notna().all():
        return False
    if pd.api.types.is_integer_dtype(s) and s.nunique() == n_rows:
        return (s.is_monotonic_increasing or s.is_monotonic_decreasing) and (s.max() - s.min() == n_rows - 1)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique() == n_rows
    return False

feature_cols = [c for c in feature_cols if not key_like(c)]
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in set(numeric_cols)]
n_numeric, n_categorical = len(numeric_cols), len(categorical_cols)

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str).str.strip()
        if vals.isin(placeholders).any() or (vals.str.len() == 0).any():
            ph_cols += 1
print(f"placeholders {sorted(placeholders)} in {ph_cols} text columns")

groups = [("all", df)]
if len(targets) == 1 and targets[0] in df.columns:
    t = targets[0]
    classes = list(pd.unique(df[t].dropna()))
    if len(classes) <= 10:
        groups += [(f"class={v}", df[df[t] == v]) for v in classes]

n_groups = len(groups)
max_rows = 50
max_features = max(1, max_rows // n_groups)
selected = []
if numeric_cols:
    selected += numeric_cols[:max(1, max_features // 2)]
if categorical_cols:
    selected += categorical_cols[:max(0, max_features - len(selected))]
for c in feature_cols:
    if len(selected) >= max_features:
        break
    if c not in selected:
        selected.append(c)
selected = selected[:max_features]
numeric_set = set(numeric_cols)

rows = []
for c in selected:
    typ = "numeric" if c in numeric_set else "categorical"
    for gname, gdf in groups:
        s = gdf[c]
        non = s.dropna()
        row = {
            "variable": c, "type": typ, "group": gname,
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_missing": int(s.isna().sum()), "n_levels": np.nan, "top_share": np.nan,
        }
        if typ == "numeric":
            if len(non):
                row.update(mean=non.mean(), sd=non.std(), min=non.min(), median=non.median(), max=non.max())
        else:
            row["n_levels"] = non.nunique()
            if len(non):
                row["top_share"] = non.value_counts(normalize=True).iloc[0]
        rows.append(row)

out = pd.DataFrame(rows, columns=[
    "variable", "type", "group", "mean", "sd", "min", "median", "max",
    "n_missing", "n_levels", "top_share",
])

def sig4(x):
    if pd.isna(x):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(f"{float(x):.4g}")
    return x

for c in ["mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_share"]:
    out[c] = out[c].map(sig4)

mlca_runtime.save_table(out, "distributions.csv", caption="Distribution summary by feature and target class")
mlca_runtime.record_fact_candidate("n_numeric", int(n_numeric), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", int(n_categorical), source="distributions.csv")
print(f"rows={len(df)} features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
print(f"distributions_rows={len(out)} groups={n_groups}")
