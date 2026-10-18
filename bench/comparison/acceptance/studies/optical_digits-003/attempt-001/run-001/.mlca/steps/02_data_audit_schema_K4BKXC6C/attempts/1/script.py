import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["optical_digits_uci"])

required = ["task", "targets", "excluded_columns"]
missing_keys = [k for k in required if k not in protocol]
if missing_keys:
    raise ValueError(f"Missing protocol keys {missing_keys}; available: {list(protocol.keys())}")

# placeholder scan for text columns
placeholders = {"?", "-", "NA", "N/A", ""}
n_placeholder_cols = 0
for c in df.columns:
    s = df[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        v = s.astype(str).str.strip()
        if v.isin(placeholders).any():
            n_placeholder_cols += 1
print(f"placeholder scan: {n_placeholder_cols} text columns contain placeholder tokens")

n = len(df)
rows = []
for c in df.columns:
    n_missing = int(df[c].isna().sum())
    rows.append({
        "name": c,
        "dtype": str(df[c].dtype),
        "n_unique": int(df[c].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / n, 4) if n else 0.0,
    })
schema = pd.DataFrame(rows)

mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

n_rows = int(len(df))
n_columns = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing_total={missing_total}")
