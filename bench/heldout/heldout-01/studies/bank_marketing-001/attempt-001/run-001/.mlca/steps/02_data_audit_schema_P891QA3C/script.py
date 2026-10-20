import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["bank_marketing_uci"])

n_rows = int(df.shape[0])
records = []
for col in df.columns:
    series = df[col]
    n_missing = int(series.isna().sum())
    pct_missing = 0.0 if n_rows == 0 else float(f"{100.0 * n_missing / n_rows:.4g}")
    records.append({
        "name": str(col),
        "dtype": str(series.dtype),
        "n_unique": int(series.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(
    records,
    columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"],
)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={df.shape[1]} missing={missing_total}")
