import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["rna_her2"])
protocol = json.loads(inputs["_protocol"].read_text())

for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing protocol key '{k}'. Available top-level keys: {list(protocol.keys())}")

n = len(df)
rows = []
for col in df.columns:
    s = df[col]
    nmiss = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": nmiss,
        "pct_missing": float(round(100.0 * nmiss / n, 4)) if n else 0.0,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(schema)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n} cols={len(schema)} missing_total={total_missing}")
