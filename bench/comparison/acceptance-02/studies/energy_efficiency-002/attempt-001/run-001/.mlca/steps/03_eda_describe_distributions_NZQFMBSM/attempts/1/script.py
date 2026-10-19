import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["energy_efficiency_uci"])

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column", None)

def key_like(col):
    s = df[col]
    n = len(s)
    if n == 0 or s.nunique(dropna=True) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return True
    return False

feature_cols = [
    c for c in df.columns
    if c not in targets and c not in excluded and c != group and not key_like(c)
]

numeric_cols = []
categorical_cols = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

rows = []
for c in numeric_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append({
        "variable": c,
        "type": "numeric",
        "n": int(s.notna().sum()),
        "n_missing": int(s.isna().sum()),
        "mean": s.mean(),
        "sd": s.std(ddof=1),
        "min": s.min(),
        "median": s.median(),
        "max": s.max(),
        "n_levels": np.nan,
        "top_level": np.nan,
        "top_level_share": np.nan,
    })

for c in categorical_cols:
    s = df[c]
    n = int(s.notna().sum())
    vc = s.dropna().value_counts()
    top = vc.index[0] if len(vc) else np.nan
    share = float(vc.iloc[0] / n) if n and len(vc) else np.nan
    rows.append({
        "variable": c,
        "type": "categorical",
        "n": n,
        "n_missing": int(s.isna().sum()),
        "mean": np.nan,
        "sd": np.nan,
        "min": np.nan,
        "median": np.nan,
        "max": np.nan,
        "n_levels": int(len(vc)),
        "top_level": str(top) if pd.notna(top) else np.nan,
        "top_level_share": share,
    })

dist = pd.DataFrame(rows)
dist = dist[[
    "variable", "type", "n", "n_missing", "mean", "sd", "min", "median",
    "max", "n_levels", "top_level", "top_level_share"
]].head(50)

for col in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    dist[col] = pd.to_numeric(dist[col], errors="coerce").map(
        lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan
    )

mlca_runtime.save_table(dist, "distributions.csv", "Distribution summaries by feature")

n_numeric = len(numeric_cols)
n_categorical = len(categorical_cols)
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

tokens = {"?", "-", "NA", "N/A", ""}
bad_cols = []
for c in df.columns:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str)
        if vals.str.strip().isin(tokens).any():
            bad_cols.append(c)

print(f"features: {len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
print(f"placeholders: {len(bad_cols)} text column(s) contain tokens such as ?, -, NA")
print("saved distributions.csv")
