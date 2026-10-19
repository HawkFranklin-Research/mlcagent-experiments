import mlca_runtime
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["energy_efficiency_uci"])
n_rows, n_cols = df.shape

def sig4(x):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = (n_missing / n_rows * 100.0) if n_rows else np.nan
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": sig4(pct_missing),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", "Column schema summary")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_cols), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"shape={df.shape}")
print(f"columns summarized={len(schema)}")
print(f"missing.total={missing_total}")
