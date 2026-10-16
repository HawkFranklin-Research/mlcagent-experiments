import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing {key} in _protocol. Available keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
task = protocol["task"]
if "primary_metric" not in task:
    raise KeyError(f"Missing primary_metric in _protocol['task']. Available keys: {list(task.keys())}")
primary_metric = task["primary_metric"]

n_rows = len(df)
n_columns = len(df.columns)
records = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    n_unique = int(df[col].nunique(dropna=True))
    pct_missing = round(n_missing / n_rows * 100, 4) if n_rows else 0.0
    records.append({
        "name": str(col),
        "dtype": str(df[col].dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

mlca_runtime.save_json(records, "schema.json", kind="table")
missing_total = sum(r["n_missing"] for r in records)
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing={missing_total}")
