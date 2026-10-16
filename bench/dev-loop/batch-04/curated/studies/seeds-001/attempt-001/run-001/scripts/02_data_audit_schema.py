import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["seeds_uci"])

rows = []
for col in df.columns:
    s = df[col]
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": int(s.isna().sum()),
        "pct_missing": round(float(s.isna().mean() * 100), 4),
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

n_rows = int(df.shape[0])
n_columns = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

placeholders = {"?", "-", "NA", "N/A", ""}
found = {}
for col in df.select_dtypes(include=["object", "string"]).columns:
    vals = df[col].dropna().astype(str).str.strip()
    hits = set(vals[vals.isin(placeholders)].unique())
    if hits:
        found[col] = hits
if found:
    print(f"placeholders found in {len(found)} columns: {found}")
else:
    print("no placeholder tokens found")
print(f"rows={n_rows} cols={n_columns} missing={missing_total}")
