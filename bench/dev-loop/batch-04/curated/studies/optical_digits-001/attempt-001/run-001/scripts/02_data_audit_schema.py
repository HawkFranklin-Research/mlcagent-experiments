import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["optical_digits_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

ph = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in df.select_dtypes(include=['object', 'string']).columns:
    s = df[c].astype(str).str.strip()
    if s.isin(ph).any():
        ph_cols.append(c)
print(f"placeholders found in {len(ph_cols)} columns")

rows = []
for c in df.columns:
    s = df[c]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(n_missing / len(df) * 100, 4) if len(df) else 0.0
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

n_rows = df.shape[0]
n_columns = df.shape[1]
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows}, cols={n_columns}, missing={missing_total}")
