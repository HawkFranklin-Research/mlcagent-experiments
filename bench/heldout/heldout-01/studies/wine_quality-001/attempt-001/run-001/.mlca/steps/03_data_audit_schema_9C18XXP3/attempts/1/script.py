import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
df = pd.read_parquet(ins["wine_quality_uci"])
_ = json.loads(ins["_protocol"].read_text())

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n = len(df)
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / n, 4) if n else float("nan"),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

n_rows = int(len(df))
n_columns = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing_total={missing_total}")
