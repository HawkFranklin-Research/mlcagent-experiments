import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["seeds_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

rows = []
for col in df.columns:
    s = df[col]
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": int(s.isna().sum()),
        "pct_missing": round(float(s.isna().mean()) * 100.0, 4),
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")

n_rows = int(len(df))
n_columns = int(df.shape[1])
missing_total = int(df.isna().sum().sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, unit="rows", source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, unit="columns", source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, unit="cells", source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing_cells={missing_total}")
