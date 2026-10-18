import json

import pandas as pd

import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["heart_disease_cleveland_uci"])
protocol = json.loads(paths["_protocol"].read_text())

n_rows = int(len(df))
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append(
        {
            "name": col,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": n_missing,
            "pct_missing": round(100.0 * n_missing / n_rows, 4) if n_rows else 0.0,
        }
    )
schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
schema = schema.head(50)
mlca_runtime.save_table(schema, "schema.json", "Per-column schema summary")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={df.shape[1]} missing_total={missing_total}")
