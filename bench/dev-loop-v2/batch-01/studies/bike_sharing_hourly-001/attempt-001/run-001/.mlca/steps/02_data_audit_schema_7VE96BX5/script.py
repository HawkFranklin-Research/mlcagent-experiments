import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])
protocol = json.loads(inputs["_protocol"].read_text())
if "task" not in protocol:
    raise KeyError(f"Missing task; available top-level keys: {list(protocol)}")

rows = []
for col in df.columns:
    s = df[col]
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": int(s.isna().sum()),
        "pct_missing": float(s.isna().mean() * 100),
    })

schema = pd.DataFrame(rows)
schema["pct_missing"] = schema["pct_missing"].round(4)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

n_rows = int(len(df))
n_columns = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"shape={df.shape}")
print(f"missing.total={missing_total}")
