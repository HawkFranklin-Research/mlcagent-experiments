import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["online_retail_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key '{key}'. Available top-level keys: {sorted(protocol.keys())}")

text_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
placeholder_cols = 0
placeholder_tokens = set()
for c in text_cols:
    s = df[c].astype("string").str.strip()
    mask = s.isin(["?", "-", "NA", "N/A"]) | s.eq("")
    if mask.any():
        placeholder_cols += 1
        placeholder_tokens.update(s[mask].dropna().unique().tolist())

records = []
for c in df.columns:
    n_missing = int(df[c].isna().sum())
    records.append({
        "name": str(c),
        "dtype": str(df[c].dtype),
        "n_unique": int(df[c].nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / max(len(df), 1), 4),
    })

mlca_runtime.save_json(records, "schema.json", kind="table")

total_missing = sum(r["n_missing"] for r in records)
mlca_runtime.record_fact_candidate("n_rows", int(len(df)), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(total_missing), source="schema.json")

print(f"shape=({len(df)}, {df.shape[1]})")
print(f"missing.total={total_missing}")
if placeholder_cols:
    print(f"placeholders={sorted(placeholder_tokens)} in {placeholder_cols} text columns")
else:
    print("placeholders=none in text columns")
