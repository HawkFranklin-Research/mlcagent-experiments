import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"_protocol missing '{key}'; available keys: {sorted(protocol)}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]

n_rows = int(len(df))
rows = []
for col in df.columns:
    s = df[col]
    miss = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": miss,
        "pct_missing": float(round(100.0 * miss / n_rows, 4)) if n_rows else float("nan"),
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_json(rows, "schema.json")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"targets={targets} excluded={excluded}")
print(f"total missing={total_missing}")
