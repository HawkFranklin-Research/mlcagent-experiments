import mlca_runtime
import json
import pandas as pd
import numpy as np


def _round_sig(value, sig=4):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(v) or np.isinf(v) or v == 0:
        return v
    return round(v, sig - int(np.floor(np.log10(abs(v)))) - 1)


inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["abalone_uci"])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

rows = []
for col in numeric_cols:
    s = df[col]
    rows.append({
        "column": col,
        "type": "numeric",
        "mean": _round_sig(s.mean()),
        "sd": _round_sig(s.std()),
        "min": _round_sig(s.min()),
        "median": _round_sig(s.median()),
        "max": _round_sig(s.max()),
        "n_missing": int(s.isna().sum()),
        "n_levels": np.nan,
        "top_level_share": np.nan,
    })

for col in categorical_cols:
    s = df[col]
    n_levels = int(s.nunique(dropna=True))
    top_share = float(s.value_counts(normalize=True, dropna=True).iloc[0]) if n_levels > 0 else 0.0
    rows.append({
        "column": col,
        "type": "categorical",
        "mean": np.nan,
        "sd": np.nan,
        "min": np.nan,
        "median": np.nan,
        "max": np.nan,
        "n_missing": int(s.isna().sum()),
        "n_levels": n_levels,
        "top_level_share": _round_sig(top_share),
    })

summary_df = pd.DataFrame(rows, columns=[
    "column", "type", "mean", "sd", "min", "median", "max",
    "n_missing", "n_levels", "top_level_share",
])

mlca_runtime.save_table(summary_df, "distributions.csv", caption="EDA distribution summary")

mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

print(f"Rows: {len(df)}, Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
