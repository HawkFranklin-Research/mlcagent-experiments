import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["online_retail_uci"])

placeholders = {"?", "-", "NA", "N/A", ""}
found = {}
for c in df.columns:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        s = df[c].astype("string").str.strip()
        mask = s.isin(["?", "-", "NA", "N/A"]) | (s == "")
        n = int(mask.sum())
        if n:
            found[c] = n
print(f"placeholder tokens in {len(found)} columns: {list(found)[:5]}")

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
missing_total = int(df.isna().sum().sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} cols={n_cols} missing={missing_total}")
