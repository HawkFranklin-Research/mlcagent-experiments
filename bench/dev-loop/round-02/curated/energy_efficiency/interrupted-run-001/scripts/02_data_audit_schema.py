import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)

# Get input paths
inputs = mlca_runtime.inputs()
data_path = inputs["energy_efficiency_uci"]
protocol_path = inputs["_protocol"]

# Read data and protocol
df = pd.read_parquet(data_path)
protocol = json.loads(protocol_path.read_text())

# Build schema table: one row per column
rows = []
for col in df.columns:
    series = df[col]
    n_missing = int(series.isna().sum())
    n_unique = int(series.nunique(dropna=True))
    pct_missing = round(100.0 * n_missing / len(df), 4)
    rows.append({
        "name": col,
        "dtype": str(series.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing
    })

schema_df = pd.DataFrame(rows)

# Save schema.json as table
mlca_runtime.save_table(schema_df, "schema.json", caption="Column schema: name, dtype, n_unique, n_missing, pct_missing")

# Record facts from schema.json
n_rows = int(len(df))
n_columns = int(len(df.columns))
missing_total = int(schema_df["n_missing"].sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, unit="rows", source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, unit="columns", source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, unit="cells", source="schema.json")

# Print brief summary
print(f"Rows: {n_rows}, Columns: {n_columns}, Missing total: {missing_total}")
