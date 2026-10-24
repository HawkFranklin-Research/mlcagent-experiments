import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["discovery"])
json.loads(inputs["_protocol"].read_text())

n_rows = int(len(df))
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / n_rows, 4) if n_rows else float("nan"),
    })

schema = pd.DataFrame.from_records(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, unit="rows", source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), unit="columns", source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, unit="cells", source="schema.json")

print(f"shape={df.shape[0]}x{df.shape[1]} missing_total={missing_total}")
print(f"columns_profiled={len(schema)}")
