import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["covertype_uci"])

rows = []
for col in df.columns:
    s = df[col]
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": int(s.isna().sum()),
        "pct_missing": round(float(s.isna().mean() * 100), 4),
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")

n_rows = int(df.shape[0])
n_columns = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing_total={missing_total}")
