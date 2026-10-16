import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["online_retail_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_cols = df.shape[1]
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct = (n_missing / n_rows * 100) if n_rows else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": float(f"{pct:.4g}")
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} cols={n_cols} missing={missing_total}")
