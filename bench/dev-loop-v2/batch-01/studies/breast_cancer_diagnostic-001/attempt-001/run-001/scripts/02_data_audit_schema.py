import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows = int(df.shape[0])
n_columns = int(df.shape[1])

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = float(100.0 * n_missing / n_rows) if n_rows else float("nan")
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": round(pct_missing, 4) if pd.notna(pct_missing) else None,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for col in df.select_dtypes(include=["object", "string", "category"]).columns:
    vals = df[col].dropna().astype(str).str.strip()
    if vals.isin(placeholders).any():
        ph_cols.append(str(col))

print(f"Schema: {n_rows} rows x {n_columns} columns; missing total {missing_total}")
print(f"Placeholder tokens found in {len(ph_cols)} text columns")
