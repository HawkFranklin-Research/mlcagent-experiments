import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["gli85_glioma_openml"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_cols = len(df.columns)
n_unique = df.nunique(dropna=True)
n_missing = df.isna().sum()

schema = pd.DataFrame({
    "name": df.columns.astype(str),
    "dtype": df.dtypes.astype(str).values,
    "n_unique": n_unique.values,
    "n_missing": n_missing.values,
})
schema["n_unique"] = schema["n_unique"].astype(int)
schema["n_missing"] = schema["n_missing"].astype(int)
schema["pct_missing"] = (
    (100.0 * schema["n_missing"] / n_rows).round(4) if n_rows else 0.0
)

mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_cols}")
print(f"missing_total={missing_total}")
