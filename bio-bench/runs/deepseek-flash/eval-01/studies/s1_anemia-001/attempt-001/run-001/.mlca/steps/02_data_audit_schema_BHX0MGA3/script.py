import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["anemia"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing _protocol[{key!r}]; available keys: {sorted(protocol)}")

n = len(df)
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(n_missing / n * 100), 4) if n else None,
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")
print(f"rows={df.shape[0]} cols={df.shape[1]} missing_total={int(schema['n_missing'].sum())}")
