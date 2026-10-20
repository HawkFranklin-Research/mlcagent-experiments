import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["wholesale_customers_uci"])

n_rows = int(df.shape[0])
n_cols = int(df.shape[1])

rows = []
for col in df.columns:
    s = df[col]
    n_unique = int(s.nunique(dropna=True))
    n_missing = int(s.isna().sum())
    pct_missing = float(n_missing / n_rows * 100) if n_rows else 0.0
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": round(pct_missing, 4),
    })

mlca_runtime.save_json(rows, "schema.json", kind="table")
total_missing = int(sum(r["n_missing"] for r in rows))

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"shape={df.shape} cols={n_cols} missing_total={total_missing}")
