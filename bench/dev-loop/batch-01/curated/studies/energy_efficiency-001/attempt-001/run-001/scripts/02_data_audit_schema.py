import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["energy_efficiency_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

placeholders = {"?", "-", "NA", "N/A"}
placeholder_cols = []
for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
        s = df[col].dropna().astype(str)
        if s.str.strip().isin(placeholders).any() or s.str.strip().eq("").any():
            placeholder_cols.append(col)
print(f"placeholders: {len(placeholder_cols)} columns" + (f" ({', '.join(placeholder_cols[:5])})" if placeholder_cols else ""))

n_rows = len(df)
n_cols = df.shape[1]
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = round(n_missing / n_rows * 100, 4) if n_rows else 0.0
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
print(f"shape: {n_rows} rows x {n_cols} columns")
print(f"saved schema.json with {len(schema)} rows")
