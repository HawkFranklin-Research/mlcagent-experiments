import json
import pandas as pd
import mlca_runtime

inputs = mlca_runtime.inputs()
mlca_runtime.set_seed(42)

_ = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["seeds_uci"])

n_rows = int(df.shape[0])
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=False))
    pct_missing = round(100.0 * n_missing / n_rows, 4) if n_rows else float("nan")
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(schema)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={len(schema)} missing={missing_total}")
