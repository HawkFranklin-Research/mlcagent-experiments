import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["heart_disease_cleveland_uci"])
_ = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_columns = df.shape[1]

def sig4(x):
    if pd.isna(x):
        return x
    return float(f"{x:.4g}")

rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    pct_missing = 100.0 * n_missing / n_rows if n_rows else float("nan")
    rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": sig4(pct_missing),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
missing_total = int(schema["n_missing"].sum())

mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"shape=({n_rows}, {n_columns})")
print(f"missing_total={missing_total}")
print("saved schema.json")
