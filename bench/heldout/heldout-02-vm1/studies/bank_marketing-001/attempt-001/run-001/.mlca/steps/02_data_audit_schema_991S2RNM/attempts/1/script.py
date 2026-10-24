import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["bank_marketing_uci"])

n_rows = len(df)
rows = []
for col in df.columns:
    s = df[col]
    missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": missing,
        "pct_missing": round(missing / n_rows, 4) if n_rows else 0.0,
    })

schema = pd.DataFrame(rows)
if len(schema) > 50:
    schema = schema.iloc[:50]

mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", len(df.columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={len(df.columns)} missing_total={missing_total}")
