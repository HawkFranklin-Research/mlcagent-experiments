import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["rna_pr"])
_protocol = json.loads(paths["_protocol"].read_text())

records = []
n_rows = int(df.shape[0])
for i, col in enumerate(df.columns):
    s = df.iloc[:, i]
    n_missing = int(s.isna().sum())
    pct_missing = float(f"{100.0 * n_missing / n_rows:.4g}") if n_rows else None
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

mlca_runtime.save_json(records, "schema.json", kind="table")
n_columns = int(df.shape[1])
missing_total = int(sum(r["n_missing"] for r in records))
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} columns={n_columns}")
print(f"missing_total={missing_total}")
