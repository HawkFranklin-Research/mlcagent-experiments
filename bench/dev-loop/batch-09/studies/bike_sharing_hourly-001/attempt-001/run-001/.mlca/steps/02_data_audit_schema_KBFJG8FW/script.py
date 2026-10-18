import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["bike_sharing_hourly_uci"])
protocol = json.loads(paths["_protocol"].read_text())

def need(obj, key):
    if key not in obj:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(obj.keys())}")
    return obj[key]

targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
task = need(protocol, "task")
_ = need(task, "primary_metric")

n_rows, n_columns = df.shape
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct = float(f"{(100.0 * n_missing / n_rows):.4g}") if n_rows else float("nan")
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"shape=({n_rows}, {n_columns})")
print(f"missing.total={total_missing}")
print(f"saved schema.json rows={len(schema)}")
