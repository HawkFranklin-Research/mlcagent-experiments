import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["proteome_pr"])

n_rows, n_columns = df.shape
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = round(100.0 * n_missing / n_rows, 4) if n_rows else 0.0
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

mlca_runtime.save_json(rows, "schema.json", kind="table")

total_missing = sum(r["n_missing"] for r in rows)
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"shape=({n_rows}, {n_columns})")
print(f"missing.total={total_missing}")
