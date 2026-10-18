import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["anuran_calls_uci"])
protocol = json.loads(paths["_protocol"].read_text())

n_rows = len(df)
n_columns = df.shape[1]
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = float(f"{100.0 * n_missing / n_rows:.4g}") if n_rows else 0.0
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
missing_total = int(schema["n_missing"].sum()) if len(schema) else 0

mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(missing_total), source="schema.json")
print(f"rows={n_rows} cols={n_columns} missing={missing_total}")
