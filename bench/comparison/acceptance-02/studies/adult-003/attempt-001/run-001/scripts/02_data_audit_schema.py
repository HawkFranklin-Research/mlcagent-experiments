import mlca_runtime
import json
import math
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

df = pd.read_parquet(paths["adult_census_uci"])
_ = json.loads(paths["_protocol"].read_text())

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return float(x)
    return float(round(x, sig - int(math.floor(math.log10(abs(x)))) - 1))

n_rows = len(df)
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct = (n_missing / n_rows * 100.0) if n_rows else float("nan")
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round_sig(pct),
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), unit="rows", source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), unit="columns", source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, unit="cells", source="schema.json")

print(f"rows={n_rows} columns={df.shape[1]}")
print(f"missing.total={missing_total}")
