import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["wholesale_customers_uci"])

n_rows = int(len(df))
n_cols = int(df.shape[1])
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct = float(n_missing / n_rows * 100) if n_rows else float("nan")
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": round(pct, 4),
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} columns={n_cols} missing={total_missing}")
print("saved schema.json")
