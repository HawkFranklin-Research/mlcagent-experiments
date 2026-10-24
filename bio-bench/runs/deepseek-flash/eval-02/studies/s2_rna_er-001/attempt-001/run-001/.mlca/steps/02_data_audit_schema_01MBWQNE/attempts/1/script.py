import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["rna_er"])

n_rows = int(df.shape[0])
n_cols = int(df.shape[1])

schema = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    pct_missing = round(100.0 * n_missing / n_rows, 4) if n_rows else 0.0
    schema.append({
        "name": str(col),
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

mlca_runtime.save_json(schema, "schema.json", kind="table")

missing_total = int(sum(row["n_missing"] for row in schema))
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, unit="cells", source="schema.json")

print(f"rows={n_rows} cols={n_cols} missing_total={missing_total}")
