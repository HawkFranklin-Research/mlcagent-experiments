import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns"]:
    if key not in proto:
        raise KeyError(f"Missing key '{key}' in _protocol. Available top-level keys: {list(proto.keys())}")

df = pd.read_parquet(inputs["adult_census_uci"])

rows = []
for col in df.columns:
    s = df[col]
    n_unique = int(s.nunique(dropna=True))
    n_missing = int(s.isna().sum())
    pct_missing = float(n_missing / len(df) * 100) if len(df) else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })
schema = pd.DataFrame(rows)

placeholders = {"?", "-", "NA", "N/A", ""}
found_tokens = set()
ph_cols = []
ws_cols = []
for col in df.columns:
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        raw = df[col].dropna().astype(str)
        stripped = raw.str.strip()
        tokens = set(stripped[stripped.isin(placeholders)].unique())
        if tokens:
            found_tokens.update(tokens)
            ph_cols.append(col)
        if (raw != stripped).any():
            ws_cols.append(col)

if found_tokens:
    print(f"Placeholders {sorted(found_tokens)} in {len(ph_cols)} cols; whitespace in {len(ws_cols)} cols")
else:
    print(f"No placeholder tokens; whitespace in {len(ws_cols)} cols")

mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

mlca_runtime.record_fact_candidate("n_rows", int(len(df)), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(df.columns)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")

print(f"rows={len(df)} cols={len(df.columns)} missing_total={int(schema['n_missing'].sum())}")
