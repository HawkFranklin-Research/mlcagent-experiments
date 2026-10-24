import json

import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["discovery"])

n_rows = int(len(df))
missing = df.isna().sum()
unique = df.nunique(dropna=True)

rows = []
for col in df.columns:
    n_miss = int(missing[col])
    rows.append({
        "name": str(col),
        "dtype": str(df[col].dtype),
        "n_unique": int(unique[col]),
        "n_missing": n_miss,
        "pct_missing": round(100.0 * n_miss / n_rows, 4) if n_rows else 0.0,
    })

mlca_runtime.save_json(rows, "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(missing.sum()), source="schema.json")
print(f"rows={n_rows} cols={df.shape[1]} missing={int(missing.sum())}")
