import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)

# Load input paths
inputs = mlca_runtime.inputs()
df_path = inputs['heart_disease_cleveland_uci']
protocol_path = inputs['_protocol']

# Read the dataset
df = pd.read_parquet(df_path)

# Read protocol JSON for completeness (not strictly needed for schema, but read as per instructions)
with open(protocol_path, 'r') as f:
    protocol = json.load(f)

# Calculate dataset level statistics
n_rows, n_columns = df.shape
total_missing = int(df.isna().sum().sum())

# Generate schema rows: one row per column
schema_rows = []
for col in df.columns:
    col_series = df[col]
    n_missing = int(col_series.isna().sum())
    n_unique = int(col_series.nunique())
    pct_missing = round((n_missing / n_rows * 100), 4) if n_rows > 0 else 0.0
    schema_rows.append({
        'name': col,
        'dtype': str(col_series.dtype),
        'n_unique': n_unique,
        'n_missing': n_missing,
        'pct_missing': pct_missing
    })

schema_df = pd.DataFrame(schema_rows, columns=['name', 'dtype', 'n_unique', 'n_missing', 'pct_missing'])

# Save schema.json as a table output
mlca_runtime.save_table(schema_df, 'schema.json', caption='Data schema summary')

# Record required facts
mlca_runtime.record_fact_candidate('n_rows', n_rows, source='schema.json')
mlca_runtime.record_fact_candidate('n_columns', n_columns, source='schema.json')
mlca_runtime.record_fact_candidate('missing.total', total_missing, source='schema.json')

# Print minimal diagnostic lines
print(f"Dataset shape: {n_rows} rows, {n_columns} columns")
print(f"Total missing values: {total_missing}")
print("Schema saved to schema.json")
