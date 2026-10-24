import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
available = list(protocol.keys())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"missing '{key}' in _protocol; available top-level keys: {available}")

df = pd.read_parquet(inputs["wine_quality_uci"])
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / len(s), 4) if len(s) else 0.0,
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print(f"rows={df.shape[0]} cols={df.shape[1]} missing={total_missing}")
