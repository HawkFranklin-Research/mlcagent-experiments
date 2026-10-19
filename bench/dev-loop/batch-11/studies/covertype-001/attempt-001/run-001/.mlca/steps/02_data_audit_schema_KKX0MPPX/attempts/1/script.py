import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["covertype_uci"])
_ = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_cols = df.shape[1]

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = (n_missing / n_rows * 100) if n_rows else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(pct_missing, 4),
    })

schema_df = pd.DataFrame(rows)
total_missing = int(schema_df["n_missing"].sum())

mlca_runtime.save_table(schema_df, "schema.json", caption="Column schema summary")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"Rows: {n_rows}, Columns: {n_cols}")
print(f"Total missing: {total_missing}")
print("Schema saved to schema.json")
