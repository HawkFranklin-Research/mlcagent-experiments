import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["seeds_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

rows = []
for col in df.columns:
    s = df[col]
    n_unique = int(s.nunique(dropna=True))
    n_missing = int(s.isna().sum())
    pct_missing = (n_missing / len(df) * 100) if len(df) else 0.0
    pct_missing = float(f"{pct_missing:.4g}")
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })
schema = pd.DataFrame(rows)

text_cols = df.select_dtypes(include=["object", "string"]).columns
found = set()
for col in text_cols:
    stripped = df[col].astype(str).str.strip()
    if stripped.isin(["?", "-", "NA", "N/A", ""]).any():
        found.add(col)
if found:
    print(f"Placeholders found in {len(found)} text columns: {', '.join(found)}")
else:
    print("No placeholder tokens found in text columns.")

mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

n_rows = len(df)
n_columns = len(df.columns)
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"n_rows={n_rows}, n_columns={n_columns}, missing.total={missing_total}")
