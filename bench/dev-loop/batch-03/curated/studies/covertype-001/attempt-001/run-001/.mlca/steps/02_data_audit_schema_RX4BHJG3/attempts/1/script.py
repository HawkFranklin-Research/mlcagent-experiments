import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["covertype_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows, n_columns = df.shape

placeholders = {"?", "-", "NA", "N/A", ""}
text_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) in ("string", "str")]
found_placeholders = set()
cols_with_placeholders = 0
for col in text_cols:
    s = df[col].dropna().astype(str).str.strip()
    mask = s.isin(placeholders)
    if mask.any():
        cols_with_placeholders += 1
        found_placeholders.update(s[mask].unique())

display_placeholders = {p for p in found_placeholders if p != ""}
if "" in found_placeholders:
    display_placeholders.add("whitespace-only")
if display_placeholders:
    print(f"Placeholders found: {', '.join(sorted(display_placeholders))} in {cols_with_placeholders} columns")
else:
    print("No placeholder tokens found.")

rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    pct = 0.0 if n_rows == 0 else (n_missing / n_rows * 100)
    if pct != 0:
        pct = float(f"{pct:.4g}")
    rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": int(df[col].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct,
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"Rows: {n_rows}, Columns: {n_columns}")
print(f"Missing total: {missing_total}")
