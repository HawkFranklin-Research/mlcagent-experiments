import mlca_runtime
import json
import pandas as pd
import numpy as np

# Set seed for reproducibility
mlca_runtime.set_seed(42)

# Get input paths
inputs = mlca_runtime.inputs()
data_path = inputs["breast_cancer_diagnostic_uci"]
protocol_path = inputs["_protocol"]

# Read the protocol JSON
protocol = json.loads(protocol_path.read_text())

# Read the data
df = pd.read_parquet(data_path)

# Build schema table: one row per column
schema_rows = []
for col in df.columns:
    n_unique = df[col].nunique()
    n_missing = df[col].isna().sum()
    pct_missing = round((n_missing / len(df)) * 100, 4)
    schema_rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing
    })

schema_df = pd.DataFrame(schema_rows)

# Save schema.json
mlca_runtime.save_json(schema_df.to_dict(orient="records"), "schema.json")

# Record facts from schema.json
n_rows = len(df)
n_columns = len(df.columns)
missing_total = int(df.isna().sum().sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

# Print summary
print(f"Rows: {n_rows}")
print(f"Columns: {n_columns}")
print(f"Total missing: {missing_total}")
