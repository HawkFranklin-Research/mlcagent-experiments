import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

df = pd.read_parquet(inputs["online_retail_uci"])
_ = json.loads(inputs["_protocol"].read_text())

n_rows = int(len(df))
n_cols = int(df.shape[1])

schema_rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = round(100.0 * n_missing / n_rows, 4) if n_rows else 0.0
    schema_rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

total_missing = int(df.isna().sum().sum())

mlca_runtime.save_json(schema_rows, "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} cols={n_cols} missing={total_missing}")
