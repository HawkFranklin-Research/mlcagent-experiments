import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise KeyError(f"missing protocol key {k!r}; available: {sorted(proto.keys())}")

df = pd.read_parquet(paths["wine_quality_uci"])

rows = []
tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
ph_total = 0
for col in df.columns:
    s = df[col]
    is_text = s.dtype == object or pd.api.types.is_string_dtype(s)
    n_ph = 0
    if is_text:
        t = s.dropna().astype(str).str.strip()
        mask = t.isin(tokens).sum()
        n_ph = int(mask.item() if hasattr(mask, "item") else mask)
        has_ws = int((s.dropna().astype(str) != s.dropna().astype(str).str.strip()).sum())
    if n_ph:
        ph_cols += 1
        ph_total += n_ph
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": int(s.isna().sum()),
        "pct_missing": round(float(s.isna().mean()) * 100.0, 4),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), unit="rows", source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), unit="columns", source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, unit="cells", source="schema.json")

print(f"shape={df.shape}")
print(f"placeholder tokens found in {ph_cols} text columns ({ph_total} values)")
print(f"total_missing={total_missing}")
