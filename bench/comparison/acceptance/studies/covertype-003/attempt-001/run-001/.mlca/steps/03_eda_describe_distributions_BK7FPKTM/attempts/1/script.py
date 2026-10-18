import json
import re
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

need = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in need if k not in protocol]
if missing:
    raise ValueError(f"Missing protocol keys {missing}; available keys: {sorted(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
task = protocol["task"]

df = pd.read_parquet(inputs["covertype_uci"])

def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)

def is_key_like(s, name):
    if is_text(s):
        return s.nunique(dropna=True) == s.notna().sum() and s.notna().sum() > 1
    lname = str(name).lower()
    if pd.api.types.is_integer_dtype(s) and any(x in lname for x in ["id", "key", "code"]):
        return s.nunique(dropna=True) == s.notna().sum() and s.notna().sum() > 1
    return False

feature_cols = [c for c in df.columns if c not in set(targets) | set(excluded) | ({group_col} if group_col else set())]
feature_cols = [c for c in feature_cols if not is_key_like(df[c], c)]

placeholders = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in df.columns:
    if is_text(df[c]):
        s = df[c].astype("string")
        stripped = s.str.strip()
        mask = stripped.isin(placeholders) | (s.notna() & stripped.eq(""))
        if mask.any():
            ph_cols.append(c)

def round_sig(x, sig=4):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")

def is_binary(s):
    return pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s) and s.nunique(dropna=True) <= 2

family_cols = {}
individual_cols = []
for c in feature_cols:
    m = re.match(r"^(Soil_Type|Wilderness_Area)_?\d+$", c)
    if m:
        family_cols.setdefault(re.sub(r"_?\d+$", "", c), []).append(c)
    else:
        individual_cols.append(c)

rows = []
for fam, cols in family_cols.items():
    shares, levels = [], []
    for c in cols:
        vc = df[c].value_counts(normalize=True, dropna=True)
        if len(vc):
            shares.append(vc.max())
            levels.append(len(vc))
    rows.append({
        "variable": fam + "_*", "type": "categorical", "n_columns": len(cols),
        "n_missing": int(sum(df[c].isna().sum() for c in cols)),
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
        "n_levels": max(levels) if levels else np.nan,
        "top_share": float(np.mean(shares)) if shares else np.nan,
        "level": np.nan,
    })

for c in individual_cols:
    s = df[c]
    if is_binary(s):
        typ = "categorical"
    elif pd.api.types.is_numeric_dtype(s):
        typ = "numeric"
    else:
        typ = "categorical"
    row = {"variable": c, "type": typ, "n_columns": 1, "n_missing": int(s.isna().sum()),
           "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
           "n_levels": np.nan, "top_share": np.nan, "level": np.nan}
    if typ == "numeric":
        row.update({"mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(), "max": s.max()})
    else:
        vc = s.value_counts(normalize=True, dropna=True)
        row["n_levels"] = int(s.nunique(dropna=True))
        row["top_share"] = float(vc.max()) if len(vc) else np.nan
    rows.append(row)

for t in targets:
    vc = df[t].value_counts(dropna=False)
    for level, count in vc.items():
        rows.append({
            "variable": t, "type": "target_class", "n_columns": 1, "n_missing": np.nan,
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": np.nan, "top_share": float(count / len(df)),
            "level": "NaN" if pd.isna(level) else str(level),
        })

dist = pd.DataFrame(rows)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[c] = dist[c].map(round_sig)
dist = dist.head(50)

mlca_runtime.save_table(dist, "distributions.csv", "Feature and target distribution summary")
n_num = int(dist.loc[dist["type"] == "numeric", "n_columns"].sum())
n_cat = int(dist.loc[dist["type"] == "categorical", "n_columns"].sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")

print(f"feature columns: {len(feature_cols)}")
print(f"summary rows: {len(dist)}")
print(f"placeholder columns: {len(ph_cols)}")
print(f"n_numeric={n_num}, n_categorical={n_cat}")
