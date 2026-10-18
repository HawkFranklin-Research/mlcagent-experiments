import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
_ = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["heart_disease_cleveland_uci"])

rows = []
for c in df.columns:
    s = df[c]
    n_miss = int(s.isna().sum())
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_miss,
        "pct_missing": round(float(n_miss) * 100.0 / max(len(s), 1), 4),
    })
schema = pd.DataFrame(rows)

mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"shape={df.shape}")
print(f"total_missing={total_missing}")
print(f"cols_in_schema={len(schema)}")
