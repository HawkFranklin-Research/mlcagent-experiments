import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["seeds_uci"])

n_rows = len(df)
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct = 0.0 if n_rows == 0 else 100.0 * n_missing / n_rows
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=False)),
        "n_missing": n_missing,
        "pct_missing": float(f"{pct:.4g}"),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"shape={df.shape}")
print(f"missing_total={missing_total}")
print("saved=schema.json")
