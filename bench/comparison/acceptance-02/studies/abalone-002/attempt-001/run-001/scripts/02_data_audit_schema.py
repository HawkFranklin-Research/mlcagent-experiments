import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

df = pd.read_parquet(inputs["abalone_uci"])

PLACEHOLDERS = {"?", "-", "NA", "N/A", ""}


def placeholder_hits(s):
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        vals = s.dropna().unique()
        return {v.strip() for v in vals if isinstance(v, str) and v.strip() in PLACEHOLDERS}
    return set()


rows = []
ph_cols = 0
ph_tokens = set()
for c in df.columns:
    s = df[c]
    hits = placeholder_hits(s)
    if hits:
        ph_cols += 1
        ph_tokens |= hits
    n_missing = int(s.isna().sum())
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / max(len(s), 1), 4),
    })

schema = pd.DataFrame(rows).head(50).reset_index(drop=True)
for col in ["n_unique", "n_missing", "pct_missing"]:
    schema[col] = schema[col].apply(lambda v: round(float(v), 4))

mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary.")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"shape={df.shape}")
print(f"placeholders found: {sorted(ph_tokens)} in {ph_cols} columns")
print(f"missing.total={total_missing}")
