import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

df = pd.read_parquet(inputs["covertype_uci"])
n_rows = len(df)

placeholder_tokens = {"?", "-", "NA", "N/A"}
ph_cols = []
for c in df.columns:
    s = df[c]
    if s.dtype == object or str(s.dtype) in ("string", "category"):
        ss = s.astype("string")
        stripped = ss.str.strip()
        mask = stripped.isin(placeholder_tokens) | (stripped == "") | ss.str.isspace().fillna(False)
        if mask.any():
            ph_cols.append(c)
print(f"Placeholder tokens in {len(ph_cols)} text columns: {ph_cols[:10]}")

rows = []
for c in df.columns:
    s = df[c]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = (n_missing / n_rows * 100) if n_rows else np.nan
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema_df = pd.DataFrame(rows)
schema_df["pct_missing"] = schema_df["pct_missing"].apply(
    lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan
)

out_path = mlca_runtime.save_table(
    schema_df, "schema.json", caption="Column schema: name, dtype, n_unique, n_missing, pct_missing"
)

mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(df.columns)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema_df["n_missing"].sum()), source="schema.json")

print(f"Rows: {n_rows}, Columns: {len(df.columns)}")
print(f"Total missing: {int(schema_df['n_missing'].sum())}")
print(f"Saved {out_path}")
