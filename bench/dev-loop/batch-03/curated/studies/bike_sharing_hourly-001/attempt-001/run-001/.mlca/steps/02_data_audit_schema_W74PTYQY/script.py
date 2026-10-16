import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = round(n_missing / len(df) * 100, 4) if len(df) else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(df.isna().sum().sum()), source="schema.json")

print(f"rows={df.shape[0]} cols={df.shape[1]} missing={df.isna().sum().sum()}")
print(f"schema rows={len(schema)}")
