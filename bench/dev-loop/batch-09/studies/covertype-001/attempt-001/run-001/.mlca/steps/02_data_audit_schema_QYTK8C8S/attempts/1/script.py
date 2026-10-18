import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["covertype_uci"])
_ = json.loads(inputs["_protocol"].read_text())

n_rows = int(len(df))
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct = 100.0 * n_missing / n_rows if n_rows else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": float(f"{pct:.4g}"),
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json")

n_columns = int(schema.shape[0])
total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print(f"rows={n_rows} columns={n_columns} missing={total_missing}")
