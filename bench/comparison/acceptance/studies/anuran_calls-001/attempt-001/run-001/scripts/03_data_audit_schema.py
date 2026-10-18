import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
protocol = json.loads(ins["_protocol"].read_text())

df = pd.read_parquet(ins["anuran_calls_uci"])

rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(n_missing) / len(df) * 100.0, 4) if len(df) else float("nan"),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(len(df)), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={len(df)} cols={df.shape[1]} missing_total={missing_total}")
print(f"columns_with_missing={int((schema['n_missing'] > 0).sum())}")
