import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
_ = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["optical_digits_uci"])

rows = []
for col in df.columns:
    s = df[col]
    n_miss = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_miss,
        "pct_missing": round(100.0 * n_miss / len(df), 4) if len(df) else 0.0,
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema", kind="table")

mlca_runtime.record_fact_candidate("n_rows", int(len(df)), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(df.columns)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")

print(f"rows={len(df)} cols={len(df.columns)} missing={int(schema['n_missing'].sum())}")
