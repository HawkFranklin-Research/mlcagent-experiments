import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns"):
    if key not in proto:
        raise KeyError(f"missing '{key}' in _protocol; available keys: {list(proto)}")

df = pd.read_parquet(inputs["covertype_uci"])
n_rows = int(len(df))

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append(
        {
            "name": col,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": n_missing,
            "pct_missing": round(100.0 * n_missing / n_rows, 3) if n_rows else 0.0,
        }
    )

schema = pd.DataFrame(rows)[["name", "dtype", "n_unique", "n_missing", "pct_missing"]]
mlca_runtime.save_table(schema, "schema.json", caption="Column schema: dtype, cardinality, missingness")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print(f"rows={n_rows} cols={df.shape[1]} missing_total={total_missing}")
