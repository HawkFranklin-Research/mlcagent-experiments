import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys: {missing}. Available top-level keys: {list(protocol.keys())}")

df = pd.read_parquet(inputs["anuran_calls_uci"])
n_rows, n_columns = df.shape

def sig4(x):
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": sig4(n_missing / n_rows * 100) if n_rows else 0.0,
    })

schema_df = pd.DataFrame(rows)
mlca_runtime.save_table(schema_df, "schema.json", caption="Column schema")

missing_total = int(schema_df["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns}")
print(f"missing_total={missing_total}")
print("schema saved to schema.json")
