import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["steel_plates_faults_uci"])

n = len(df)
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(n_missing) / n * 100, 4) if n else float("nan"),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n} cols={df.shape[1]} missing={missing_total}")
