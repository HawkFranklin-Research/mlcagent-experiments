import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])
_ = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_cols = df.shape[1]
missing = df.isna().sum()

records = []
for col in df.columns:
    n_missing = int(missing[col])
    records.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(n_missing / n_rows * 100, 4) if n_rows else float("nan"),
    })

schema = pd.DataFrame(records, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} cols={n_cols} missing_total={total_missing}")
