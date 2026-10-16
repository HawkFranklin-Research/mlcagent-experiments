import json
import mlca_runtime
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["optical_digits_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])

placeholders = {"?", "-", "NA", "N/A", ""}
found = {}
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
        s = df[c].astype(str).str.strip()
        hits = int(s.isin(placeholders).sum())
        if hits:
            found[c] = hits
print(f"placeholder columns: {len(found)}")

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
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

n_rows = int(df.shape[0])
n_cols = int(df.shape[1])
total_missing = int(schema["n_missing"].sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} cols={n_cols} missing={total_missing}")
