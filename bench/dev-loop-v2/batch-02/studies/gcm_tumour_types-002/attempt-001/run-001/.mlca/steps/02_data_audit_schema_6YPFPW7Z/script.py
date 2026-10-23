import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["gcm_tumour_types_openml"])

n_rows = int(df.shape[0])
n_columns = int(df.shape[1])
rows = []
for i, col in enumerate(df.columns):
    s = df.iloc[:, i]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct = (n_missing / n_rows * 100.0) if n_rows else 0.0
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": float(f"{pct:.4g}"),
    })

total_missing = int(sum(r["n_missing"] for r in rows))
mlca_runtime.save_json(rows, "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print(f"rows={n_rows} cols={n_columns} missing={total_missing}")
