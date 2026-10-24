import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["proteome_her2"])

n_rows = int(len(df))
records = []
for i, col in enumerate(df.columns):
    s = df.iloc[:, i]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct = 100.0 * n_missing / n_rows if n_rows else 0.0
    pct = float(f"{pct:.4g}")
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct,
    })

mlca_runtime.save_json(records, "schema.json", kind="table")
missing_total = int(sum(r["n_missing"] for r in records))
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", len(records), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} cols={len(records)} missing_total={missing_total}")
