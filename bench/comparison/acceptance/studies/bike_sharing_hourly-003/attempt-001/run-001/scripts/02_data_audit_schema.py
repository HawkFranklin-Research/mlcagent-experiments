import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])

def sig4(x):
    return float(f"{x:.4g}") if x != 0 else 0.0

n_rows, n_columns = int(df.shape[0]), int(df.shape[1])
rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    pct_missing = (100.0 * n_missing / n_rows) if n_rows else 0.0
    rows.append({
        "name": str(col),
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": sig4(pct_missing),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
total_missing = int(schema["n_missing"].sum())

mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} columns={n_columns} missing_total={total_missing}")
