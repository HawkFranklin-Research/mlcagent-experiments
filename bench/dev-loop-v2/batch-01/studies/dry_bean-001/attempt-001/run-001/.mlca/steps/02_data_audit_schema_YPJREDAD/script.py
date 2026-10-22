import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["dry_bean_uci"])

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in df.columns:
    if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
        s = df[c].dropna().astype(str).str.strip()
        if s.isin(placeholders).any():
            ph_cols.append(c)

rows = []
for c in df.columns:
    nm = int(df[c].isna().sum())
    rows.append({
        "name": c,
        "dtype": str(df[c].dtype),
        "n_unique": int(df[c].nunique(dropna=True)),
        "n_missing": nm,
        "pct_missing": round(100.0 * nm / len(df), 4) if len(df) else 0.0,
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema of frozen table")

mlca_runtime.record_fact_candidate("n_rows", int(len(df)), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")

print(f"shape={df.shape}")
print(f"missing_total={int(schema['n_missing'].sum())}")
print(f"placeholder_columns={len(ph_cols)} tokens={sorted(placeholders)}")
