import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["heart_disease_cleveland_uci"])
_ = json.loads(paths["_protocol"].read_text())

n_rows = len(df)
n_cols = df.shape[1]
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = (100.0 * n_missing / n_rows) if n_rows else 0.0
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(pct_missing), 4),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} columns={n_cols} missing_total={total_missing}")
