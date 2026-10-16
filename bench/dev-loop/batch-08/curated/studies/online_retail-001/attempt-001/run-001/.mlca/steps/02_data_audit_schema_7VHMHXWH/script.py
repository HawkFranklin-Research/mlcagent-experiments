import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

df = pd.read_parquet(inputs["online_retail_uci"])
_ = json.loads(inputs["_protocol"].read_text())

n_rows = int(df.shape[0])
n_columns = int(df.shape[1])

def sig4(x):
    if pd.isna(x):
        return x
    return float(f"{x:.4g}")

schema = pd.DataFrame({
    "name": df.columns.astype(str),
    "dtype": [str(t) for t in df.dtypes],
    "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    "n_missing": [int(df[c].isna().sum()) for c in df.columns],
})
if n_rows:
    schema["pct_missing"] = (schema["n_missing"] / n_rows * 100).apply(sig4)
else:
    schema["pct_missing"] = float("nan")

mlca_runtime.save_table(schema, "schema.json", "Column schema")
missing_total = int(schema["n_missing"].sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns}")
print(f"missing.total={missing_total}")
