import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["discovery"])
n_rows, n_cols = df.shape

schema = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = float(f"{(n_missing / n_rows * 100) if n_rows else 0.0:.4g}")
    schema.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

mlca_runtime.save_json(schema, "schema.json", kind="table")
missing_total = int(sum(r["n_missing"] for r in schema))
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_cols), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
found_tokens = set()
for col in df.columns:
    s = df[col]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        vals = s.dropna().astype(str).str.strip()
        mask = vals.isin(placeholders)
        if mask.any():
            found_tokens.update(vals[mask].unique())
            placeholder_cols.append(str(col))

print(f"rows={n_rows} cols={n_cols} missing_total={missing_total}")
print(f"placeholders={sorted(found_tokens)} columns={len(placeholder_cols)}")
