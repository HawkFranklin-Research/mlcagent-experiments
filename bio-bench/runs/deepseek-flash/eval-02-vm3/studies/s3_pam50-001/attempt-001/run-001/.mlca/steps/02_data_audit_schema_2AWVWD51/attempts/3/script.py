import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["discovery"])
json.loads(inputs["_protocol"].read_text())

n_rows = int(len(df))
n_cols = int(df.shape[1])

rows = []
for i in range(n_cols):
    s = df.iloc[:, i]
    n_missing = int(s.isna().sum())
    try:
        n_unique = int(s.nunique(dropna=True))
    except TypeError:
        n_unique = int(s.astype(str).nunique(dropna=True))
    pct = round(100.0 * n_missing / n_rows, 4) if n_rows else float("nan")
    rows.append({
        "name": str(df.columns[i]),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct,
    })

schema = pd.DataFrame.from_records(
    rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"]
)
mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, unit="rows", source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, unit="columns", source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, unit="cells", source="schema.json")

print(f"shape={n_rows}x{n_cols} missing_total={missing_total}")
print(f"columns_profiled={len(schema)}")
