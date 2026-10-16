import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["online_retail_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_columns = len(df.columns)

rows = []
for col in df.columns:
    s = df[col]
    n_unique = int(s.nunique(dropna=True))
    n_missing = int(s.isna().sum())
    pct_missing = round(n_missing / n_rows * 100, 4) if n_rows else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema_df = pd.DataFrame(rows)
mlca_runtime.save_table(schema_df, "schema.json", caption="Column schema")

missing_total = int(schema_df["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

placeholders = {"?", "-", "NA", "N/A", ""}
found = set()
cols_with = 0
for col in df.select_dtypes(include=["object", "string"]).columns:
    s = df[col].dropna().astype(str)
    if s.empty:
        continue
    stripped = s.str.strip()
    mask = stripped.isin(placeholders) | (stripped == "")
    if mask.any():
        cols_with += 1
        found.update(stripped[mask].unique())

if found:
    print(f"Placeholders found: {sorted(found)} in {cols_with} columns")
else:
    print("No placeholders found")

print(f"Rows: {n_rows}, Columns: {n_columns}")
print(f"Missing total: {missing_total}")
