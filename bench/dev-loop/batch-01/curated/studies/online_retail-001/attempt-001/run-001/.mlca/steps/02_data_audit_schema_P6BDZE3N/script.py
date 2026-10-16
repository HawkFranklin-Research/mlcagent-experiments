import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["online_retail_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_columns = df.shape[1]

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = float(f"{n_missing / n_rows * 100:.4g}") if n_rows > 0 else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing
    })

schema_df = pd.DataFrame(rows)
mlca_runtime.save_table(schema_df, "schema.json", caption="Column schema")

total_missing = int(schema_df["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"n_rows={n_rows}")
print(f"n_columns={n_columns}")
print(f"missing.total={total_missing}")
