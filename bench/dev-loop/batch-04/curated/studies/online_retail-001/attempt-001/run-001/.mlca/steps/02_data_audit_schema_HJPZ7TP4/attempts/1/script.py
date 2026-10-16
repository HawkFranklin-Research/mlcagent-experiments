import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["online_retail_uci"])
_protocol = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
n_cols = df.shape[1]

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = round(100.0 * n_missing / n_rows, 4) if n_rows else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })
schema = pd.DataFrame(rows)

mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

placeholders = {"?", "-", "NA", "N/A", ""}
found = set()
cols_with_ph = 0
cols_with_ws = 0
for col in df.columns:
    s = df[col]
    if s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype) == "str":
        s = s.dropna().astype(str)
        stripped = s.str.strip()
        if (s != stripped).any():
            cols_with_ws += 1
        mask = stripped.isin(placeholders)
        if mask.any():
            cols_with_ph += 1
            found.update(stripped[mask].unique())
print(f"Placeholders found: {sorted(found)} in {cols_with_ph} text columns; whitespace in {cols_with_ws}")
print(f"rows={n_rows} cols={n_cols} missing={missing_total}")
