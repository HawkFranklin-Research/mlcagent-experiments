import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["wholesale_customers_uci"])

rows = []
for c in df.columns:
    s = df[c]
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": int(s.isna().sum()),
        "pct_missing": round(float(s.isna().mean() * 100.0), 4),
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print(f"rows={df.shape[0]} cols={df.shape[1]} missing_total={total_missing}")
