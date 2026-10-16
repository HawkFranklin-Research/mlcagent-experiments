import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = round(n_missing / len(df) * 100, 4) if len(df) else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", len(df), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", df.shape[1], source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={len(df)} cols={df.shape[1]} missing_total={missing_total}")
