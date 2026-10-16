import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
for k in required:
    if k not in proto:
        raise KeyError(f"Missing key {k} in _protocol. Available keys: {list(proto.keys())}")

targets = proto["targets"]
excluded = proto["excluded_columns"]
group_col = proto["group_column"]
df = pd.read_parquet(inputs["covertype_uci"])

# Define feature columns once (RL-0007)
all_cols = list(df.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]
key_like = [c for c in feature_cols if df[c].dtype == object and df[c].nunique() == len(df)]
feature_cols = [c for c in feature_cols if c not in key_like]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

rows = []
for c in numeric_cols:
    s = df[c]
    rows.append({
        "variable": c, "type": "numeric", "group": "overall",
        "n": len(s), "n_missing": int(s.isna().sum()),
        "mean": s.mean(), "sd": s.std(), "min": s.min(),
        "median": s.median(), "max": s.max(),
        "n_levels": np.nan, "top_share": np.nan
    })

if categorical_cols:
    top_shares = []
    for c in categorical_cols:
        vc = df[c].value_counts(dropna=False)
        top_shares.append(vc.iloc[0] / len(df))
    rows.append({
        "variable": "categorical_indicators", "type": "categorical", "group": "overall",
        "n": len(df), "n_missing": int(df[categorical_cols].isna().sum().sum()),
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
        "n_levels": 2, "top_share": float(np.mean(top_shares))
    })

target = targets[0]
if target in df.columns:
    vc = df[target].value_counts(dropna=False)
    for cls, cnt in vc.items():
        rows.append({
            "variable": target, "type": "target_class", "group": str(cls),
            "n": int(cnt), "n_missing": 0,
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": np.nan, "top_share": float(cnt / len(df))
        })

dist = pd.DataFrame(rows)
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    if col in dist.columns:
        dist[col] = dist[col].apply(lambda x: round_sig(x) if pd.notna(x) else x)

mlca_runtime.save_table(dist, "distributions.csv", caption="Distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

placeholders = ['?', '-', 'NA', 'N/A']
ph_cols = []
for c in df.columns:
    if df[c].dtype == object:
        s = df[c].astype(str).str.strip()
        if (s.isin(placeholders) | (s == '')).any():
            ph_cols.append(c)
print(f"placeholders found: {placeholders} in {len(ph_cols)} columns")
print(f"n_numeric={len(numeric_cols)} n_categorical={len(categorical_cols)}")
print(f"distributions rows={len(dist)}")
print(f"target classes={df[target].nunique() if target in df.columns else 0}")
