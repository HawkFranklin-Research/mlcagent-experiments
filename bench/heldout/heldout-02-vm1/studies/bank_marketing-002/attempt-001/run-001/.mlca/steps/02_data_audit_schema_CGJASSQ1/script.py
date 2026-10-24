import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["bank_marketing_uci"])
protocol = json.loads(paths["_protocol"].read_text())

n_rows = len(df)
rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / n_rows, 4) if n_rows else 0.0,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")

print(f"rows={n_rows} columns={df.shape[1]}")
print(f"missing_total={int(schema['n_missing'].sum())}")
print("saved schema.json")
