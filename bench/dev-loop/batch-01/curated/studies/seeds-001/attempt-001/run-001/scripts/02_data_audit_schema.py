import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["seeds_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

def need(key):
    if key not in protocol:
        raise KeyError(f"Missing _protocol key '{key}'. Available top-level keys: {sorted(protocol.keys())}")
    return protocol[key]

task = need("task")
targets = need("targets")
excluded = need("excluded_columns")
group = need("group_column")
primary_metric = need("primary_metric") if "primary_metric" in protocol else None

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = (n_missing / len(df) * 100.0) if len(df) else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(pct_missing, 4),
    })
schema = pd.DataFrame(rows)

mlca_runtime.save_table(schema, "schema.json", caption="Column schema")
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")

print(f"rows={df.shape[0]} cols={df.shape[1]}")
print(f"missing.total={int(schema['n_missing'].sum())}")
