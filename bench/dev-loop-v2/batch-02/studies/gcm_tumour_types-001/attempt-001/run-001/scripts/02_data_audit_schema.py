import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

_protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["gcm_tumour_types_openml"])

n_rows = int(df.shape[0])
n_cols = int(df.shape[1])

n_missing = df.isna().sum()
schema = pd.DataFrame({
    "name": df.columns.astype(str),
    "dtype": df.dtypes.astype(str).to_numpy(),
    "n_unique": df.nunique(dropna=True).to_numpy(),
    "n_missing": n_missing.to_numpy(),
})
schema["pct_missing"] = (
    schema["n_missing"] / n_rows * 100
).round(4) if n_rows else 0.0

mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} cols={n_cols} missing={total_missing}")
print(f"schema_rows={len(schema)}")
