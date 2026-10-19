import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key}; available keys: {list(protocol)}")

df = pd.read_parquet(paths["covertype_uci"])
n_rows = len(df)
n_columns = len(df.columns)

records = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(n_missing / n_rows * 100, 4) if n_rows else None,
    })

missing_total = int(sum(r["n_missing"] for r in records))
mlca_runtime.save_json(records, "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} cols={n_columns} missing={missing_total}")
