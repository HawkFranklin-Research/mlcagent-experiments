import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["smk187_airway_openml"])

n = len(df)
rows = []
for c in df.columns:
    nmiss = int(df[c].isna().sum())
    rows.append({
        "name": c,
        "dtype": str(df[c].dtype),
        "n_unique": int(df[c].nunique(dropna=True)),
        "n_missing": nmiss,
        "pct_missing": round(100.0 * nmiss / n, 4) if n else float("nan"),
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={df.shape[0]} cols={df.shape[1]}")
print(f"missing_total={total_missing}")
print(f"schema_rows={len(schema)}")
