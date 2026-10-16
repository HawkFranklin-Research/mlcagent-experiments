import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["optical_digits_uci"])

if "targets" not in protocol:
    raise ValueError(f"Missing 'targets' in _protocol. Available keys: {list(protocol.keys())}")
if "excluded_columns" not in protocol:
    raise ValueError(f"Missing 'excluded_columns' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column", None)

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]
key_like = [c for c in feature_cols if df[c].dtype == 'object' and df[c].nunique() == len(df)]
feature_cols = [c for c in feature_cols if c not in key_like]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

if numeric_cols:
    vals = df[numeric_cols].to_numpy(dtype=float, na_value=np.nan).ravel()
    mean = np.nanmean(vals)
    sd = np.nanstd(vals, ddof=1)
    min_val = np.nanmin(vals)
    median = np.nanmedian(vals)
    max_val = np.nanmax(vals)
    n_missing = int(np.isnan(vals).sum())
else:
    mean = sd = min_val = median = max_val = n_missing = np.nan

n_levels = 0
top_level_share = np.nan

summary = pd.DataFrame([
    {"type": "numeric", "n_columns": len(numeric_cols), "mean": mean, "sd": sd, "min": min_val,
     "median": median, "max": max_val, "n_missing": n_missing, "n_levels": np.nan, "top_level_share": np.nan},
    {"type": "categorical", "n_columns": len(categorical_cols), "mean": np.nan, "sd": np.nan, "min": np.nan,
     "median": np.nan, "max": np.nan, "n_missing": np.nan, "n_levels": n_levels, "top_level_share": top_level_share}
])

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

for col in ["mean", "sd", "min", "median", "max", "n_missing", "n_levels", "top_level_share"]:
    summary[col] = summary[col].apply(lambda x: round_sig(x) if pd.notna(x) else x)

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distributions summary")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
print("Saved distributions.csv")
