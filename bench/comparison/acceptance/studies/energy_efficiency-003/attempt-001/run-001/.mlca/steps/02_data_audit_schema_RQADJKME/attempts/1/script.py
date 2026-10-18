import json
import math

import pandas as pd

import mlca_runtime

mlca_runtime.set_seed(42)

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["energy_efficiency_uci"])

n_rows = int(len(df))
n_cols = int(df.shape[1])


def round_sig(x, sig=4):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")


records = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    n_unique = int(df[col].nunique(dropna=True))
    pct_missing = None if n_rows == 0 else (n_missing / n_rows * 100.0)
    records.append(
        {
            "name": str(col),
            "dtype": str(df[col].dtype),
            "n_unique": n_unique,
            "n_missing": n_missing,
            "pct_missing": round_sig(pct_missing),
        }
    )

mlca_runtime.save_json(records, "schema.json", kind="table")

total_missing = int(sum(r["n_missing"] for r in records))
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows}, cols={n_cols}, missing={total_missing}")
