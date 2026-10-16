import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["anuran_calls_uci"])
_ = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_columns = df.shape[1]

schema = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    schema.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(n_missing / n_rows * 100, 4) if n_rows else 0.0,
    })

missing_total = int(sum(row["n_missing"] for row in schema))

mlca_runtime.save_json(schema, "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing={missing_total}")
