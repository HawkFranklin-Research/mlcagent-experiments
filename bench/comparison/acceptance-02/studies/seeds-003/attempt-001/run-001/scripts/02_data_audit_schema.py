import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["seeds_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

allowed = {"targets", "excluded_columns", "group_column", "task"}
missing = allowed - set(protocol)
if missing:
    raise KeyError(f"_protocol missing {sorted(missing)}; available keys: {sorted(protocol)}")

rows = []
for col in df.columns:
    s = df[col]
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": int(s.isna().sum()),
        "pct_missing": float(round(s.isna().mean() * 100, 4)),
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_json(rows, "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={df.shape[0]} cols={df.shape[1]}")
print(f"missing_total={total_missing}")
print("saved schema.json")
