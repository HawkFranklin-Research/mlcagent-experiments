import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["abalone_uci"])
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
        "pct_missing": round(100.0 * n_missing / len(df), 4)
    })

schema_df = pd.DataFrame(rows)
mlca_runtime.save_table(schema_df, "schema.json", caption="Column schema")

n_rows = int(len(df))
n_columns = int(df.shape[1])
missing_total = int(df.isna().sum().sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, unit="rows", source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, unit="columns", source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, unit="cells", source="schema.json")

print(f"Rows: {n_rows}, Columns: {n_columns}, Missing total: {missing_total}")
