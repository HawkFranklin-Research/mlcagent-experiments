import mlca_runtime
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
data_path = paths["breast_cancer_diagnostic_uci"]

if data_path.suffix.lower() == ".parquet":
    df = pd.read_parquet(data_path)
elif data_path.suffix.lower() in (".csv", ".tsv"):
    sep = "\t" if data_path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(data_path, sep=sep)
else:
    raise ValueError(f"Unsupported data format: {data_path.suffix}")

n_rows, n_cols = df.shape

def r4(x: float) -> float:
    if pd.isna(x):
        return float("nan")
    return float(f"{x:.4g}")

records = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = 100.0 * n_missing / n_rows if n_rows else float("nan")
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": r4(pct_missing),
    })

schema = pd.DataFrame(records, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
missing_total = int(schema["n_missing"].sum())

mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_cols), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} columns={n_cols} missing_total={missing_total}")
print(f"saved schema.json with {len(schema)} rows")
