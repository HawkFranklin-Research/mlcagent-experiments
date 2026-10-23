import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["smk187_airway_openml"])
protocol = json.loads(inputs["_protocol"].read_text())
n_rows, n_cols = df.shape

def sig4(x):
    if pd.isna(x):
        return x
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    rows.append({
        "name": str(col),
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": sig4((n_missing / n_rows * 100.0) if n_rows else float("nan")),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")
total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_cols), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print(f"rows={n_rows} columns={n_cols} schema_rows={len(schema)}")
print(f"missing.total={total_missing}")
