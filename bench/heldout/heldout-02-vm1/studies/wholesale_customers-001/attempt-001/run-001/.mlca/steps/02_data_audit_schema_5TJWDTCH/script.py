import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["wholesale_customers_uci"])

def sig4(x):
    return float(f"{x:.4g}")

rows = []
n = len(df)
for col in df.columns:
    s = df[col]
    nm = int(s.isna().sum())
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": nm,
        "pct_missing": sig4(100.0 * nm / n) if n else 0.0,
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")

mlca_runtime.record_fact_candidate("n_rows", int(n), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(schema)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")

print(f"rows={n} cols={len(schema)}")
print(f"missing_total={int(schema['n_missing'].sum())}")
print("schema.json saved")
