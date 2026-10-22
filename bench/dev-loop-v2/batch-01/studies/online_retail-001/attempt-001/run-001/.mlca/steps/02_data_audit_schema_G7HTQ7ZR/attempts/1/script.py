import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["online_retail_uci"])

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(n_missing / len(df) * 100.0, 4) if len(df) else 0.0,
    })
mlca_runtime.save_json(rows, "schema.json", kind="table")

total_missing = int(sum(r["n_missing"] for r in rows))
mlca_runtime.record_fact_candidate("n_rows", int(len(df)), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={len(df)} cols={df.shape[1]} missing_total={total_missing}")
