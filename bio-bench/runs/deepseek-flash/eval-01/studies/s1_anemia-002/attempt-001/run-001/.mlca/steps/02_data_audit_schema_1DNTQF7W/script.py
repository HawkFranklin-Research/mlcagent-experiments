import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["anemia"])
protocol = json.loads(paths["_protocol"].read_text())

n_rows, n_cols = df.shape


def sig4(x):
    if pd.isna(x):
        return float("nan")
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")


rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = sig4(100.0 * n_missing / n_rows) if n_rows else float("nan")
    rows.append(
        {
            "name": col,
            "dtype": str(s.dtype),
            "n_unique": n_unique,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
        }
    )

schema = pd.DataFrame(
    rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"]
)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_cols), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} columns={n_cols} missing_total={total_missing}")
