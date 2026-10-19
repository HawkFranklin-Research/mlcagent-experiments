import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key '{key}'; available top-level keys: {list(protocol.keys())}")
_task = protocol["task"]
if "primary_metric" not in _task:
    raise KeyError(f"Missing task key 'primary_metric'; available task keys: {list(_task.keys())}")

df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])
n_rows = int(df.shape[0])
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(n_missing) / n_rows * 100, 4) if n_rows else float("nan"),
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")
print(f"rows={n_rows} cols={df.shape[1]} missing={int(schema['n_missing'].sum())}")
