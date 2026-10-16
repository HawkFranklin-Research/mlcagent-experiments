import mlca_runtime
import json
import pandas as pd
import numpy as np

# Load inputs
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["abalone_uci"])

# Determine numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Build summary rows
rows = []

# Numeric columns summary
for col in numeric_cols:
    s = df[col]
    rows.append({
        "column": col,
        "type": "numeric",
        "mean": round(s.mean(), 4),
        "sd": round(s.std(), 4),
        "min": round(s.min(), 4),
        "median": round(s.median(), 4),
        "max": round(s.max(), 4),
        "n_missing": int(s.isna().sum())
    })

# Categorical columns summary
for col in categorical_cols:
    s = df[col]
    n_levels = s.nunique()
    top_share = round(s.value_counts(normalize=True).iloc[0], 4) if n_levels > 0 else 0.0
    rows.append({
        "column": col,
        "type": "categorical",
        "n_levels": int(n_levels),
        "top_level_share": top_share,
        "mean": "",
        "sd": "",
        "min": "",
        "median": "",
        "max": "",
        "n_missing": int(s.isna().sum())
    })

summary_df = pd.DataFrame(rows)

# Save distributions.csv
mlca_runtime.save_table(summary_df, "distributions.csv", caption="EDA summary of numeric and categorical columns")

# Record facts
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_cols), source="distributions.csv")

# Print headline
print(f"Rows: {len(df)}, Numeric cols: {len(numeric_cols)}, Categorical cols: {len(categorical_cols)}")
