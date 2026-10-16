import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["abalone_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_columns = df.shape[1]
rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    pct_missing = float(f"{100.0 * n_missing / n_rows:.4g}") if n_rows else 0.0
    rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
path = mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")
missing_total = int(schema["n_missing"].sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing={missing_total}")
print(f"schema saved: {path}")
