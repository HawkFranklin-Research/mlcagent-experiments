import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

df = pd.read_parquet(inputs["discovery"])
n_rows = int(df.shape[0])
n_cols = int(df.shape[1])

rows = []
for col in df.columns:
    s = df[col]
    n_miss = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_miss,
        "pct_missing": round(float(n_miss) / n_rows * 100.0, 4) if n_rows else 0.0,
    })

mlca_runtime.save_json(rows, "schema.json", kind="report")

missing_total = int(sum(r["n_missing"] for r in rows))
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_cols} missing={missing_total}")
