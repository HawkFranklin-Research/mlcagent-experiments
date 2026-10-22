import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
df = pd.read_parquet(inp["discovery"])

n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
rows = []
for c in df.columns:
    s = df[c]
    nm = int(s.isna().sum())
    pct = (nm / n_rows * 100.0) if n_rows else 0.0
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": nm,
        "pct_missing": float(f"{pct:.4g}"),
    })

mlca_runtime.save_json(rows, "schema.json", kind="table")

total_missing = int(df.isna().sum().sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print("rows", n_rows, "cols", n_cols, "missing", total_missing)
