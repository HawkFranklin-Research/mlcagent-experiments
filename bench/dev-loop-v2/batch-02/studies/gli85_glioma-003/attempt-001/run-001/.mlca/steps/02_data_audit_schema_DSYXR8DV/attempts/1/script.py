import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["gli85_glioma_openml"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows, n_columns = df.shape
records = []

for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = float(f"{100.0 * n_missing / n_rows:.4g}") if n_rows else 0.0
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

mlca_runtime.save_json(records, "schema.json", kind="table")

missing_total = int(sum(r["n_missing"] for r in records))
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} columns={n_columns} missing_total={missing_total}")
print("saved schema.json")
