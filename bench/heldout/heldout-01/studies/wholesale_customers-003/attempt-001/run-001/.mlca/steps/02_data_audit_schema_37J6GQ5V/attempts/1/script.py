import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["wholesale_customers_uci"])

rows = []
n_rows = len(df)
for c in df.columns:
    n_miss = int(df[c].isna().sum())
    rows.append({
        "name": c,
        "dtype": str(df[c].dtype),
        "n_unique": int(df[c].nunique(dropna=True)),
        "n_missing": n_miss,
        "pct_missing": round(100.0 * n_miss / max(n_rows, 1), 4),
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")

n_columns = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} cols={n_columns} missing_total={missing_total}")
