import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"_protocol missing required key {key!r}; available top-level keys: {sorted(protocol)}")
_ = protocol["targets"], protocol["excluded_columns"], protocol["task"]

df = pd.read_parquet(inputs["wine_quality_uci"])
n_rows, n_columns = df.shape

def sig4(x):
    if pd.isna(x):
        return x
    x = float(x)
    if x == 0:
        return 0.0
    return round(x, 4 - int(np.floor(np.log10(abs(x)))) - 1)

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct = float("nan") if n_rows == 0 else 100.0 * n_missing / n_rows
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": sig4(pct),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} columns={n_columns} missing_total={missing_total} schema_rows={len(schema)}")
