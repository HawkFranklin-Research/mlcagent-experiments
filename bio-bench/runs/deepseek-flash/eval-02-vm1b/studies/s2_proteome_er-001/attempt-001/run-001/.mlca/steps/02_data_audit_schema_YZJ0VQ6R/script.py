import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["proteome_er"])
_ = json.loads(inputs["_protocol"].read_text())

n_rows, n_cols = df.shape
records = []
total_missing = 0

for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    total_missing += n_missing
    records.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / n_rows, 4) if n_rows else float("nan"),
    })

mlca_runtime.save_json(records, "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_cols), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(total_missing), source="schema.json")
print(f"rows={n_rows} cols={n_cols} missing={total_missing}")
