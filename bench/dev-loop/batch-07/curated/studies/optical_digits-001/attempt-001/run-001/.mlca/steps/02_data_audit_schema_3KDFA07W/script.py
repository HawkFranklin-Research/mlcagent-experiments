import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["optical_digits_uci"])
_ = json.loads(inputs["_protocol"].read_text())

rows = []
n_rows = len(df)
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(n_missing / n_rows * 100, 4) if n_rows else 0.0,
    })

schema_df = pd.DataFrame(rows)
mlca_runtime.save_table(schema_df, "schema.json", caption="Column schema")

missing_total = int(schema_df["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", len(df.columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={len(df.columns)}")
print(f"missing_total={missing_total}")
