import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

df = pd.read_parquet(inputs["anemia"])

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
            "pct_missing": float(round(100.0 * s.isna().mean(), 4)) if len(s) else 0.0,
        }
    )

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
if len(schema) > 50:
    schema = schema.iloc[:50].copy()

mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")

total_missing = int(df.isna().sum().sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={df.shape[0]} cols={df.shape[1]} missing_total={total_missing}")
