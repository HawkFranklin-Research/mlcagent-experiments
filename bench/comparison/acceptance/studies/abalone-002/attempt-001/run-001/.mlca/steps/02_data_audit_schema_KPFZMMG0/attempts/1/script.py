import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise KeyError(f"missing _protocol['{k}']; available keys: {sorted(proto.keys())}")

df = pd.read_parquet(inp["abalone_uci"])

PLACEHOLDERS = {"?", "-", "NA", "N/A", "None", "null", "nan", "NaN", ""}
text_like = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
found = {}
for c in text_like:
    vals = df[c].astype("string").str.strip()
    hits = vals[vals.isin(list(PLACEHOLDERS))].dropna().unique().tolist()
    if hits:
        found[c] = sorted(hits)
tokens = sorted({t for v in found.values() for t in v})
print(f"placeholders found: {tokens} in {len(found)} column(s)")

rows = []
for c in df.columns:
    s = df[c]
    n_missing = int(s.isna().sum())
    rows.append(
        {
            "name": c,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": n_missing,
            "pct_missing": round(100.0 * n_missing / max(len(s), 1), 4),
        }
    )

schema = pd.DataFrame(rows).head(50)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema: dtype, unique counts, missingness.")

mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), unit="cells", source="schema.json")

print(f"shape={df.shape} columns={len(df.columns)} missing_cells={int(schema['n_missing'].sum())}")
