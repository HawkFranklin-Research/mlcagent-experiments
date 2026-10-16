import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

# RL-0023: detect placeholder tokens in text columns
tokens = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
        s = df[c].astype("string").str.strip()
        if s.isin(list(tokens)).any() or (s == "").any():
            ph_cols.append(c)
print(f"placeholder tokens found in {len(ph_cols)} text columns: {ph_cols[:5]}")

rows = []
for c in df.columns:
    s = df[c]
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": int(s.isna().sum()),
        "pct_missing": round(float(s.isna().mean() * 100.0), 4),
    })
schema = pd.DataFrame(rows)

mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

n_rows = int(df.shape[0])
n_columns = int(df.shape[1])
missing_total = int(df.isna().sum().sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} columns={n_columns} missing_total={missing_total}")
print(f"schema rows={len(schema)}")
