import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise ValueError(f"Missing protocol key '{key}'; available keys: {list(protocol.keys())}")

targets = set(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol["group_column"]
df = pd.read_parquet(inputs["optical_digits_uci"])

drop_cols = set(targets) | set(excluded)
if group is not None:
    drop_cols.add(group)

def is_key_like(series):
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series.dtype):
        vals = series.dropna().astype(str).str.strip()
        return len(vals) > 0 and vals.nunique() == len(vals)
    return False

feature_cols = [c for c in df.columns if c not in drop_cols and not is_key_like(df[c])]

placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s.dtype):
        vals = s.dropna().astype(str).str.strip()
        if vals.isin(placeholders).any():
            placeholder_cols.append(c)
print(f"Placeholder tokens found in {len(placeholder_cols)} text columns: {placeholder_cols[:5]}")

numeric_cols, categorical_cols = [], []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(s):
        continue
    else:
        n = len(s)
        if n > 0 and s.nunique(dropna=True) <= 0.05 * n:
            categorical_cols.append(c)

num_rows = []
for c in numeric_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    num_rows.append({
        "mean": s.mean(), "sd": s.std(), "min": s.min(),
        "median": s.median(), "max": s.max(), "n_missing": int(s.isna().sum()),
    })

cat_rows = []
for c in categorical_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s.dtype):
        s = s.astype(str).str.strip().mask(lambda x: x.isin(placeholders), np.nan)
    vc = s.value_counts(dropna=True)
    cat_rows.append({
        "n_levels": int(s.nunique(dropna=True)),
        "top_level_share": float(vc.iloc[0] / len(s)) if len(vc) and len(s) else np.nan,
        "n_missing": int(s.isna().sum()),
    })

def agg_num(rows):
    if not rows:
        return {k: np.nan for k in ["mean", "sd", "min", "median", "max", "n_missing"]}
    d = pd.DataFrame(rows)
    return {
        "mean": d["mean"].mean(), "sd": d["sd"].mean(), "min": d["min"].min(),
        "median": d["median"].median(), "max": d["max"].max(),
        "n_missing": d["n_missing"].sum(),
    }

def agg_cat(rows):
    if not rows:
        return {"n_levels": np.nan, "top_level_share": np.nan}
    d = pd.DataFrame(rows)
    return {"n_levels": d["n_levels"].sum(), "top_level_share": d["top_level_share"].mean()}

na, ca = agg_num(num_rows), agg_cat(cat_rows)
summary = pd.DataFrame([
    {"feature_type": "numeric", "n_columns": len(numeric_cols),
     "mean": na["mean"], "sd": na["sd"], "min": na["min"], "median": na["median"],
     "max": na["max"], "n_missing": na["n_missing"], "n_levels": np.nan,
     "top_level_share": np.nan},
    {"feature_type": "categorical", "n_columns": len(categorical_cols),
     "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
     "max": np.nan, "n_missing": np.nan, "n_levels": ca["n_levels"],
     "top_level_share": ca["top_level_share"]},
])

def round_sig(x, sig=4):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")

for col in ["mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_level_share"]:
    summary[col] = summary[col].apply(round_sig)
summary["n_columns"] = summary["n_columns"].astype(int)

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summary by type")

n_numeric = int(summary.loc[summary["feature_type"] == "numeric", "n_columns"].iloc[0])
n_categorical = int(summary.loc[summary["feature_type"] == "categorical", "n_columns"].iloc[0])
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Features: {len(feature_cols)} total, {n_numeric} numeric, {n_categorical} categorical")
print(f"Saved distributions.csv with {len(summary)} rows")
