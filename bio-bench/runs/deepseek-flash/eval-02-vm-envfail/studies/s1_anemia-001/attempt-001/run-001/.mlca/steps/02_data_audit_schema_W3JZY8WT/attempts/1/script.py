import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["anemia"])

n_rows = int(df.shape[0])
n_cols = int(df.shape[1])
rows = []
for col in df.columns:
    miss = int(df[col].isna().sum())
    rows.append({
        "name": str(col),
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": miss,
        "pct_missing": round(miss / n_rows * 100, 4) if n_rows else float("nan"),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

missing_total = int(schema["n_missing"].sum()) if not schema.empty else 0
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print("rows", n_rows, "cols", n_cols)
print("missing.total", missing_total)
