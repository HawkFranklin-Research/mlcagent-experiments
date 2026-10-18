import json
import mlca_runtime
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["dry_bean_uci"])
protocol = json.loads(inputs["_protocol"].read_text())
if not isinstance(protocol, dict):
    raise ValueError("_protocol must be a JSON object")

def sig4(x):
    if pd.isna(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

n_rows, n_columns = df.shape
records = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    records.append({
        "name": str(col),
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": sig4(n_missing / n_rows * 100.0) if n_rows else None,
    })

schema = pd.DataFrame(records, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
missing_total = int(schema["n_missing"].sum())
mlca_runtime.save_json(records, "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} cols={n_columns}")
print(f"schema_rows={len(schema)} saved=schema.json")
print(f"missing.total={missing_total}")
