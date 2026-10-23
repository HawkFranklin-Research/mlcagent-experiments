import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["anemia"])

rows = []
n = len(df)
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = float(f"{(n_missing / n * 100):.4g}") if n else float("nan")
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(rows, "schema.json", kind="table")

n_rows = int(len(df))
n_columns = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, unit="cells", source="schema.json")
print(f"rows={n_rows} columns={n_columns} missing.total={missing_total}")
