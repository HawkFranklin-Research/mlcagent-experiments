import json

import pandas as pd

import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])

n_rows, n_columns = df.shape
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": float(round((n_missing / n_rows * 100) if n_rows else 0.0, 4)),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns}")
print(f"missing_total={missing_total}")
print("saved schema.json")
