import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["dry_bean_uci"])

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct = 100.0 * n_missing / len(df) if len(df) else float("nan")
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": float(f"{pct:.4g}"),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

n_rows = len(df)
n_columns = df.shape[1]
missing_total = int(schema["n_missing"].sum()) if not schema.empty else 0
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} cols={n_columns} missing={missing_total}")
