import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / max(len(df), 1), 4),
    })
schema = pd.DataFrame(rows)
if len(schema) > 50:
    schema = schema.head(50)
if schema.shape[1] > 12:
    schema = schema.iloc[:, :12]

mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(len(df)), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={len(df)} cols={df.shape[1]}")
print(f"missing_total={missing_total}")
print(f"targets={protocol['targets']}")
