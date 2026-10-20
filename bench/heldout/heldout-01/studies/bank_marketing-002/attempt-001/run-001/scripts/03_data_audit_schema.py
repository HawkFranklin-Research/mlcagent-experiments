import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]

df = pd.read_parquet(paths["bank_marketing_uci"])

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(n_missing) / len(df) * 100.0, 4) if len(df) else 0.0,
    })
schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])

mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")

print(f"rows={df.shape[0]} cols={df.shape[1]} missing_total={int(schema['n_missing'].sum())}")
