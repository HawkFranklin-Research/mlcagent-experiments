import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["rna_er"])
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"missing '{key}' in _protocol; available keys: {sorted(protocol)}")

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(s.isna().mean()) * 100.0, 4),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema of the frozen RNA-seq table")

n_rows = int(df.shape[0])
n_cols = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"table shape: {n_rows} rows x {n_cols} columns")
print(f"missing cells total: {missing_total}; columns with any missing: {int((schema['n_missing'] > 0).sum())}")
