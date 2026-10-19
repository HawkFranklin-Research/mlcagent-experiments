import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"] if "group_column" in protocol else None

df = pd.read_parquet(inputs["adult_census_uci"])

rows = []
for c in df.columns:
    s = df[c]
    n_miss = int(s.isna().sum())
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_miss,
        "pct_missing": round(float(n_miss) / len(s) * 100.0, 4) if len(s) else 0.0,
    })
schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={df.shape[0]} cols={df.shape[1]}")
print(f"missing.total={total_missing}")
print(f"targets={targets} excluded={excluded}")
