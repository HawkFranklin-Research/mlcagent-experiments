import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["gli85_glioma_openml"])

n = len(df)
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct = 100.0 * n_missing / n if n else None
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": float(f"{pct:.4g}") if pct is not None else None,
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={df.shape[0]} columns={df.shape[1]} missing.total={total_missing}")
