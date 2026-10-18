import json

import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["energy_efficiency_uci"])
_ = json.loads(inputs["_protocol"].read_text())

n_rows = int(len(df))
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(n_missing / n_rows * 100.0) if n_rows else 0.0, 4),
    })

schema = pd.DataFrame(
    rows,
    columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"],
)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(schema)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} cols={len(schema)} missing={total_missing}")
