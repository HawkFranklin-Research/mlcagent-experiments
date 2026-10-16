import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_columns = df.shape[1]

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = 0.0 if n_rows == 0 else float(f"{n_missing / n_rows * 100:.4g}")
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing={missing_total}")
