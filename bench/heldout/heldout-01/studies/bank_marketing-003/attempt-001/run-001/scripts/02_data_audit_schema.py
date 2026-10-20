import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bank_marketing_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")
_ = protocol["task"]["primary_metric"]
_ = protocol["targets"]
_ = protocol["excluded_columns"]

n = len(df)
rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    pct_missing = float(f"{(n_missing / n * 100):.4g}") if n else 0.0
    rows.append({
        "name": str(col),
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

mlca_runtime.save_json(rows, "schema.json", kind="table")
total_missing = sum(r["n_missing"] for r in rows)
mlca_runtime.record_fact_candidate("n_rows", int(n), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(total_missing), source="schema.json")
print(f"rows={n} columns={df.shape[1]}")
print(f"missing_total={total_missing}")
