import mlca_runtime
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["anemia"])

rows = []
n = len(df)
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct = np.nan if n == 0 else float(round(100.0 * n_missing / n, 4))
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct,
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

n_rows = int(df.shape[0])
n_columns = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows}, columns={n_columns}")
print(f"missing_total={missing_total}")
print("schema saved: schema.json")
