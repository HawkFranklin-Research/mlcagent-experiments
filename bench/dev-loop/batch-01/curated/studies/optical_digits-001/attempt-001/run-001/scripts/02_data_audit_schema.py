import mlca_runtime
import pandas as pd
import json
from math import log10, floor

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["optical_digits_uci"])

protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing key {key} in _protocol. Available keys: {list(protocol.keys())}")

n_rows = len(df)
n_columns = len(df.columns)

def round_sig(x, sig=4):
    if x == 0:
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

schema_rows = []
for col in df.columns:
    s = df[col]
    n_unique = s.nunique(dropna=True)
    n_missing = s.isna().sum()
    pct_missing = (n_missing / n_rows * 100) if n_rows > 0 else 0.0
    pct_missing = round_sig(pct_missing, 4)
    schema_rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(n_unique),
        "n_missing": int(n_missing),
        "pct_missing": pct_missing
    })

schema_df = pd.DataFrame(schema_rows)
mlca_runtime.save_json(schema_df.to_dict(orient="records"), "schema.json", kind="table")

missing_total = int(schema_df["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"Rows: {n_rows}, Columns: {n_columns}")
print(f"Total missing: {missing_total}")
