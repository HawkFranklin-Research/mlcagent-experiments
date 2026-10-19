import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["heart_disease_cleveland_uci"])

n_rows = len(df)
n_columns = len(df.columns)

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(n_missing / n_rows * 100, 4) if n_rows else float("nan"),
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} columns={n_columns} missing_total={missing_total}")
