import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["abalone_uci"])

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")

skip = set(targets) | set(excluded) | ({group} if group else set())
features = [c for c in df.columns if c not in skip]

key_like = []
for c in features:
    s = df[c]
    s_nona = s.dropna()
    if len(s_nona) == 0:
        continue
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s):
        if s_nona.nunique() == len(s_nona):
            key_like.append(c)
features = [c for c in features if c not in key_like]

placeholder_tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
for c in features:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        if stripped.isin(placeholder_tokens).any():
            ph_cols += 1
print(f"placeholder tokens in {ph_cols} text columns")

rows = []
for c in features:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        rows.append({
            "column": c, "type": "numeric",
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_missing": int(s.isna().sum()),
            "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        top = float(vc.iloc[0] / vc.sum()) if vc.size else np.nan
        rows.append({
            "column": c, "type": "categorical",
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_missing": int(s.isna().sum()),
            "n_levels": int(vc.size), "top_share": top,
        })

summary = pd.DataFrame(rows, columns=[
    "column", "type", "mean", "sd", "min", "median", "max",
    "n_missing", "n_levels", "top_share",
])

def _round(x):
    if isinstance(x, (int, np.integer)):
        return x
    if isinstance(x, (float, np.floating)) and not pd.isna(x):
        return float(f"{x:.4g}")
    return x

for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[col] = summary[col].map(_round)

mlca_runtime.save_table(summary, "distributions.csv")

n_numeric = int((summary["type"] == "numeric").sum())
n_categorical = int((summary["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"features: {len(features)}  numeric: {n_numeric}  categorical: {n_categorical}")
print(f"rows in summary: {len(summary)}")
