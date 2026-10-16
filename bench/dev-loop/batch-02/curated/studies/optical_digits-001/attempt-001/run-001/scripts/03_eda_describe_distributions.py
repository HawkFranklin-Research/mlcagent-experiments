import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol:
    raise KeyError(f"missing targets; available keys: {list(protocol.keys())}")
if "excluded_columns" not in protocol:
    raise KeyError(f"missing excluded_columns; available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)

df = pd.read_parquet(inputs["optical_digits_uci"])

def is_key_like(s: pd.Series) -> bool:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s) and any(k in s.name.lower() for k in ["id", "index"])
    return False

candidate_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]
features = [c for c in candidate_cols if not is_key_like(df[c])]

numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in features if c not in numeric_cols]

rows = []
if numeric_cols:
    all_vals = df[numeric_cols].to_numpy().ravel()
    all_vals = all_vals[~pd.isna(all_vals)]
    rows.append({
        "feature_type": "numeric",
        "n_features": len(numeric_cols),
        "mean": np.mean(all_vals) if len(all_vals) else np.nan,
        "sd": np.std(all_vals, ddof=1) if len(all_vals) > 1 else np.nan,
        "min": np.min(all_vals) if len(all_vals) else np.nan,
        "median": np.median(all_vals) if len(all_vals) else np.nan,
        "max": np.max(all_vals) if len(all_vals) else np.nan,
        "n_missing": int(df[numeric_cols].isna().sum().sum()),
        "n_levels": np.nan,
        "top_level_share": np.nan,
    })
else:
    rows.append({"feature_type": "numeric", "n_features": 0, "mean": np.nan, "sd": np.nan,
                 "min": np.nan, "median": np.nan, "max": np.nan, "n_missing": 0,
                 "n_levels": np.nan, "top_level_share": np.nan})

if categorical_cols:
    levels, shares = [], []
    for c in categorical_cols:
        vc = df[c].value_counts(dropna=False)
        levels.append(df[c].nunique(dropna=False))
        shares.append(vc.iloc[0] / len(df) if len(vc) > 0 else np.nan)
    rows.append({
        "feature_type": "categorical",
        "n_features": len(categorical_cols),
        "mean": np.nan,
        "sd": np.nan,
        "min": np.nan,
        "median": np.nan,
        "max": np.nan,
        "n_missing": int(df[categorical_cols].isna().sum().sum()),
        "n_levels": np.mean(levels) if levels else np.nan,
        "top_level_share": np.mean(shares) if shares else np.nan,
    })
else:
    rows.append({"feature_type": "categorical", "n_features": 0, "mean": np.nan, "sd": np.nan,
                 "min": np.nan, "median": np.nan, "max": np.nan, "n_missing": 0,
                 "n_levels": np.nan, "top_level_share": np.nan})

dist = pd.DataFrame(rows)

def round4(x):
    if isinstance(x, (int, float, np.number)) and not pd.isna(x):
        if x == 0:
            return 0.0
        return float(f"{x:.4g}")
    return x

dist = dist.map(round4)

mlca_runtime.save_table(dist, "distributions.csv", caption="Feature distribution summary")

n_numeric = int(dist.loc[dist["feature_type"] == "numeric", "n_features"].sum())
n_categorical = int(dist.loc[dist["feature_type"] == "categorical", "n_features"].sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

text_cols = [c for c in features if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
ph_cols = []
for c in text_cols:
    s = df[c].astype(str).str.strip()
    if s.isin(["?", "-", "NA", "N/A"]).any():
        ph_cols.append(c)
print(f"features: {len(features)} (numeric={n_numeric}, categorical={n_categorical})")
print(f"placeholders: {len(ph_cols)} text columns")
print(f"saved distributions.csv rows={len(dist)}")
