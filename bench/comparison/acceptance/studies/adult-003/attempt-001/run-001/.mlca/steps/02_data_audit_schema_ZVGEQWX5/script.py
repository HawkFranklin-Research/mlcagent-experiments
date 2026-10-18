import mlca_runtime
import pandas as pd
import math

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["adult_census_uci"])

n_rows = int(df.shape[0])

def round_sig(x, sig=4):
    if x == 0 or math.isnan(x):
        return x
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

records = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = round_sig(100.0 * n_missing / n_rows) if n_rows else float("nan")
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(records, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(records, "schema.json", kind="table")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, unit="rows", source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), unit="columns", source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, unit="cells", source="schema.json")

print(f"rows={n_rows}, cols={df.shape[1]}")
print(f"columns summarized={len(schema)}, missing_total={missing_total}")
