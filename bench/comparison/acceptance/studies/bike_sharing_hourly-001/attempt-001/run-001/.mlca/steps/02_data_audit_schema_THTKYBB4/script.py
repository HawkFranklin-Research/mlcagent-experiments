import json

import numpy as np
import pandas as pd

import mlca_runtime

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["bike_sharing_hourly_uci"])
protocol = json.loads(paths["_protocol"].read_text())
if not isinstance(protocol, dict):
    raise ValueError("_protocol must be a JSON object")

n_rows, n_columns = df.shape
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append(
        {
            "name": str(col),
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": n_missing,
            "pct_missing": round(float(n_missing) / n_rows * 100.0, 4) if n_rows else np.nan,
        }
    )

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema and missingness summary")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing_total={missing_total}")
