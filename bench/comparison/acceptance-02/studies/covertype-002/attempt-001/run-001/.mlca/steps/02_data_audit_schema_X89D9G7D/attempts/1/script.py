import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
top_keys = sorted(proto.keys())
for k in ("targets", "excluded_columns"):
    if k not in proto:
        raise KeyError(f"Missing required protocol key '{k}'. Available top-level keys: {top_keys}")

df = pd.read_parquet(inputs["covertype_uci"])

records = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct = round(100.0 * n_missing / len(df), 4) if len(df) else 0.0
    records.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct,
    })

schema = pd.DataFrame(records)
mlca_runtime.save_json(records, "schema.json", kind="table")

n_rows = int(len(df))
n_cols = int(len(df.columns))
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_cols} missing_total={missing_total}")
