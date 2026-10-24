import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
_ = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["rna_her2"])

n_rows = len(df)
schema = pd.DataFrame({
    "name": df.columns.astype(str),
    "dtype": df.dtypes.astype(str).to_numpy(),
    "n_unique": df.nunique(dropna=True).to_numpy(),
    "n_missing": df.isna().sum().to_numpy(),
})
schema["pct_missing"] = (schema["n_missing"] / max(n_rows, 1) * 100).round(4)

mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")

print(f"schema_rows={len(schema)} data_rows={n_rows} missing_total={int(schema['n_missing'].sum())}")
