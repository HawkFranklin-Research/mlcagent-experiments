import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["energy_efficiency_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol or "excluded_columns" not in protocol:
    raise KeyError(f"Missing protocol keys; available: {list(protocol.keys())}")

rows = []
n_rows = len(df)
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = float(f"{(100.0 * n_missing / n_rows) if n_rows else float('nan'):.4g}")
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(df.isna().sum().sum()), source="schema.json")

print(f"rows={df.shape[0]} cols={df.shape[1]} missing={int(df.isna().sum().sum())}")
