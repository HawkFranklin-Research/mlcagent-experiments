import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bank_marketing_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows, n_cols = df.shape

def sig(x):
    return float(f"{x:.4g}")

records = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = sig(100.0 * n_missing / n_rows) if n_rows else 0.0
    records.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(records, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(records, "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_cols), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"shape: {n_rows} x {n_cols}")
print(f"missing.total: {total_missing}")
print("saved schema.json")
