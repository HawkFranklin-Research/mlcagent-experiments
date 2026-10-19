import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
_ = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["covertype_uci"])

n_rows = len(df)
n_columns = df.shape[1]
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / n_rows, 4) if n_rows else 0.0,
    })
schema = pd.DataFrame(rows)
total_missing = int(df.isna().sum().sum())

mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print(f"rows={n_rows} cols={n_columns} missing={total_missing}")
