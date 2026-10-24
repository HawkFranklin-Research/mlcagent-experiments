import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["proteome_pr"])
protocol = json.loads(inputs["_protocol"].read_text())
if "targets" not in protocol:
    raise KeyError(f"_protocol keys: {sorted(protocol.keys())}")

n = len(df)
cols = ["name", "dtype", "n_unique", "n_missing", "pct_missing"]
rows = []
for c in df.columns:
    s = df[c]
    nm = int(s.isna().sum())
    rows.append([
        str(c),
        str(s.dtype),
        int(s.nunique(dropna=True)),
        nm,
        round(100.0 * nm / n, 4) if n else 0.0,
    ])

mlca_runtime.save_json({"columns": cols, "rows": rows}, "schema.json", kind="table")

total_missing = sum(r[3] for r in rows)
mlca_runtime.record_fact_candidate("n_rows", int(n), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(total_missing), unit="cells", source="schema.json")

print(f"rows={n} cols={df.shape[1]} missing_total={total_missing}")
