import json
import math
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["gcm_tumour_types_openml"])

n_rows = len(df)
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = round(float(n_missing / n_rows * 100), 4) if n_rows else float("nan")
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

placeholder_tokens = {"?", "-", "NA", "N/A", ""}
placeholder_cols = 0
found_tokens = set()
for col in df.select_dtypes(include=["object", "string"]).columns:
    values = set(df[col].dropna().astype(str).str.strip().unique())
    hits = values & placeholder_tokens
    if hits:
        placeholder_cols += 1
        found_tokens |= hits
if placeholder_cols:
    print(f"placeholders={sorted(found_tokens)} columns={placeholder_cols}")

n_columns = len(df.columns)
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing={missing_total}")
