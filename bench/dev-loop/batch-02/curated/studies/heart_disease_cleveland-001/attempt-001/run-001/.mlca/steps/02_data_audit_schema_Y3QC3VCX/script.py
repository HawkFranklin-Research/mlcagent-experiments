import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
data_path = inputs["heart_disease_cleveland_uci"]
protocol_path = inputs["_protocol"]

# Read protocol (required by rails, even if not used for this action)
protocol = json.loads(protocol_path.read_text())

df = pd.read_parquet(data_path)

n_rows = len(df)
n_columns = len(df.columns)

def round_sig(x, sig=4):
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")

schema_rows = []
for col in df.columns:
    n_unique = int(df[col].nunique(dropna=True))
    n_missing = int(df[col].isna().sum())
    pct_missing = (n_missing / n_rows * 100) if n_rows > 0 else 0.0
    schema_rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": round_sig(pct_missing)
    })

schema_df = pd.DataFrame(schema_rows)

mlca_runtime.save_json(schema_df.to_dict(orient="records"), "schema.json", kind="table")

missing_total = int(schema_df["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"n_rows: {n_rows}")
print(f"n_columns: {n_columns}")
print(f"missing.total: {missing_total}")
