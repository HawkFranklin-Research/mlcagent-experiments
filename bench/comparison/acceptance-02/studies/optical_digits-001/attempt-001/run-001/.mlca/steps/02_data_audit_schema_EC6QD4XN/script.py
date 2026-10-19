import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["optical_digits_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Protocol missing required key {key!r}; available keys: {sorted(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
n_rows = len(df)

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = 100.0 * n_missing / n_rows if n_rows else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": float(f"{pct_missing:.4g}"),
    })

schema = pd.DataFrame(
    rows,
    columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"],
)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(df.columns)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")

print(f"rows={n_rows} cols={len(df.columns)}")
print(f"missing_total={int(schema['n_missing'].sum())}")
print("saved schema.json")
