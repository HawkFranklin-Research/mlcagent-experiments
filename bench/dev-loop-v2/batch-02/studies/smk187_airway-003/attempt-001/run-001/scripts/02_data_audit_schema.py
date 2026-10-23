import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def need(obj, key, path):
    if key not in obj:
        raise KeyError(f"Missing {path}.{key}; available keys at {path}: {list(obj)}")
    return obj[key]

need(protocol, "task", "_protocol")
task = protocol["task"]
need(task, "primary_metric", "_protocol.task")
need(protocol, "targets", "_protocol")
need(protocol, "excluded_columns", "_protocol")

df = pd.read_parquet(paths["smk187_airway_openml"])

rows = []
n = len(df)
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(n_missing) / n * 100.0, 4) if n else 0.0,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(rows, "schema.json", kind="table")

n_rows = int(df.shape[0])
n_columns = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} columns={n_columns}")
print(f"missing.total={missing_total}")
print("saved schema.json")
