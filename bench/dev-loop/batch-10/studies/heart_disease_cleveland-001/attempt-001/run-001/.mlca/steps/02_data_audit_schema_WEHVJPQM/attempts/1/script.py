import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["heart_disease_cleveland_uci"])
protocol = json.loads(paths["_protocol"].read_text())

placeholders = {"?", "-", "NA", "N/A", ""}
rows = []
found = set()
ph_cols = 0
n = len(df)

for name in df.columns:
    col = df[name]
    missing = col.isna()
    if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
        s = col.astype("string")
        stripped = s.str.strip()
        ph_mask = s.notna() & stripped.isin(placeholders)
        if ph_mask.any():
            found.update(stripped[ph_mask].dropna().unique().tolist())
            ph_cols += 1
        missing = missing | ph_mask
    n_missing = int(missing.sum())
    n_unique = int(col.mask(missing).nunique(dropna=True))
    pct_missing = float(f"{(100.0 * n_missing / n if n else 0.0):.4g}")
    rows.append({
        "name": str(name),
        "dtype": str(col.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

mlca_runtime.save_json(rows, "schema.json", kind="table")
missing_total = sum(r["n_missing"] for r in rows)
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(missing_total), source="schema.json")
print(f"rows={df.shape[0]} columns={df.shape[1]} missing_total={missing_total}")
print(f"placeholders found={sorted(found)} columns={ph_cols}")
