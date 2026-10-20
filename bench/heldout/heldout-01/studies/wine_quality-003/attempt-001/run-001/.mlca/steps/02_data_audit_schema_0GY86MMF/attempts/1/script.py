import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["wine_quality_uci"])

records = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    records.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(n_missing) / max(len(df), 1) * 100.0, 4),
    })

mlca_runtime.save_json(records, "schema.json", kind="table")

n_rows = int(df.shape[0])
n_cols = int(df.shape[1])
total_missing = int(sum(r["n_missing"] for r in records))

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} cols={n_cols} total_missing={total_missing}")
print(f"schema_rows={len(records)}")
