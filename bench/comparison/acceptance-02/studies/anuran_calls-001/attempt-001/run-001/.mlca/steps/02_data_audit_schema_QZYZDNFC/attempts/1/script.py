import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["anuran_calls_uci"])

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in proto]
if missing:
    raise KeyError(f"missing protocol keys {missing}; available: {list(proto.keys())}")

# placeholder audit on text columns (report only; nothing is fitted here)
ph_tokens = {"?", "-", "NA", "N/A", ""}
ph_cols = 0
for c in df.columns:
    if df[c].dtype == object or str(df[c].dtype).startswith("str"):
        s = df[c].astype("string").str.strip()
        if s.isin(list(ph_tokens)).any():
            ph_cols += 1

n_rows = int(len(df))
rows = []
for c in df.columns:
    s = df[c]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / n_rows, 4) if n_rows else 0.0,
    })
schema = pd.DataFrame(rows)
schema_out = schema.head(50)

mlca_runtime.save_json(schema_out.to_dict(orient="records"), "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(schema)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} cols={len(schema)} missing_total={total_missing}")
print(f"placeholder_tokens_found_in_columns={ph_cols}")
print(f"schema_rows_saved={len(schema_out)}")
