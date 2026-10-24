import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["rna_er"])
_ = json.loads(paths["_protocol"].read_text())

n_rows = len(df)
records = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    try:
        n_unique = int(s.nunique(dropna=True))
    except TypeError:
        n_unique = int(s.astype(str).nunique(dropna=True))
    pct_missing = round(100.0 * n_missing / n_rows, 4) if n_rows else float("nan")
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(records, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

missing_total = int(schema["n_missing"].sum()) if len(schema) else 0
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(df.columns)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

placeholders = {"?", "-", "NA", "N/A", "na", "n/a", "N/A", ""}
found_tokens = set()
found_cols = 0
for col in df.columns:
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        stripped = df[col].dropna().astype(str).str.strip()
        mask = stripped.isin(placeholders) | (stripped == "")
        if mask.any():
            found_cols += 1
            found_tokens.update(stripped[mask].unique().tolist())

print(f"rows={n_rows} cols={len(df.columns)} missing={missing_total}")
print(f"placeholder_tokens={sorted(found_tokens)} columns={found_cols}")
