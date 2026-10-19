import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
_ = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])

rows = []
for col in df.columns:
    s = df[col]
    miss = s.isna()
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": int(miss.sum()),
        "pct_missing": float(f"{miss.mean() * 100:.4g}"),
    })
schema = pd.DataFrame(rows)

mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

n_rows = int(df.shape[0])
n_cols = int(df.shape[1])
total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"shape: {n_rows} x {n_cols}")
print(f"missing total: {total_missing}")
print(f"columns with missing: {int((schema['n_missing'] > 0).sum())}")
