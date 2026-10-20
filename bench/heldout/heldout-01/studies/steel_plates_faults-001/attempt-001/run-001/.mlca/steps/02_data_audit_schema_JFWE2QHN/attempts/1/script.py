import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["steel_plates_faults_uci"])
_ = json.loads(paths["_protocol"].read_text())

n_rows, n_cols = df.shape
if n_cols > 50:
    raise ValueError(f"schema would have {n_cols} rows, exceeding the 50-row summary limit")

rows = []
for col in df.columns:
    missing = int(df[col].isna().sum())
    pct = (missing / n_rows * 100) if n_rows else 0.0
    rows.append({
        "name": str(col),
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": missing,
        "pct_missing": float(f"{pct:.4g}"),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_cols), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_cols} missing={missing_total}")
print("saved schema.json")
