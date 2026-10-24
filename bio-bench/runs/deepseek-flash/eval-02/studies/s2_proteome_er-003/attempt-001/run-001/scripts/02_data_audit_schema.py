import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["proteome_er"])

n_rows = int(len(df))
n_columns = int(df.shape[1])

records = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = float(round(n_missing / n_rows * 100.0, 4)) if n_rows else float("nan")
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

missing_total = int(sum(r["n_missing"] for r in records))

mlca_runtime.save_json(records, "schema.json", kind="table")

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} columns={n_columns}")
print(f"missing.total={missing_total}")
print("saved schema.json")
