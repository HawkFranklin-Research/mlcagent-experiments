import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["rna_pr"])

n_rows = int(df.shape[0])
n_cols = int(df.shape[1])
total_missing = 0
records = []

for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    total_missing += n_missing
    n_unique = int(s.nunique(dropna=True))
    pct_missing = float(f"{100.0 * n_missing / n_rows:.4g}") if n_rows else float("nan")
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

mlca_runtime.save_json(records, "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(total_missing), source="schema.json")
print(f"rows={n_rows} columns={n_cols} missing={total_missing}")
