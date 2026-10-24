import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["discovery"])
protocol = json.loads(inputs["_protocol"].read_text())

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / len(df), 4) if len(df) else float("nan"),
    })
schema = pd.DataFrame(rows)

mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")
mlca_runtime.record_fact_candidate("n_rows", int(len(df)), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(df.isna().sum().sum()), source="schema.json")

print(f"rows={len(df)} cols={df.shape[1]}")
print(f"total_missing={int(df.isna().sum().sum())}")
