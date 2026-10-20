import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("task", "targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing _protocol key {key!r}; available top-level keys: {sorted(protocol)}")
if not isinstance(protocol["task"], dict) or "primary_metric" not in protocol["task"]:
    raise KeyError(
        "Missing _protocol['task']['primary_metric']; "
        f"available task keys: {sorted(protocol['task']) if isinstance(protocol['task'], dict) else type(protocol['task']).__name__}"
    )
_ = protocol["task"]["primary_metric"]
_ = protocol["targets"]
_ = protocol["excluded_columns"]

df = pd.read_parquet(inputs["wholesale_customers_uci"])
n_rows = len(df)

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = 0.0 if n_rows == 0 else 100.0 * n_missing / n_rows
    rows.append(
        {
            "name": col,
            "dtype": str(s.dtype),
            "n_unique": n_unique,
            "n_missing": n_missing,
            "pct_missing": float(f"{pct_missing:.4g}"),
        }
    )

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(schema)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} columns={len(schema)} missing_total={missing_total}")
