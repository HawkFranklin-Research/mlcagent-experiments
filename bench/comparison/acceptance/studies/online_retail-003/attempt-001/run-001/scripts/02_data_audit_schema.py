import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["online_retail_uci"])
protocol = json.loads(paths["_protocol"].read_text())

n_rows = int(len(df))
n_cols = int(df.shape[1])

rows = []
found_placeholders = set()
placeholder_cols = 0

for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = 100.0 * n_missing / n_rows if n_rows else 0.0
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": round(pct_missing, 4),
    })
    if pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str).str.strip()
        col_hit = False
        for tok in ("?", "-", "NA", "N/A"):
            if (vals == tok).any():
                found_placeholders.add(tok)
                col_hit = True
        if (vals == "").any():
            found_placeholders.add("<whitespace>")
            col_hit = True
        if col_hit:
            placeholder_cols += 1

schema_df = pd.DataFrame(rows)
mlca_runtime.save_table(schema_df, "schema.json", caption="Per-column schema summary")

missing_total = int(schema_df["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_cols} missing_total={missing_total}")
print(f"placeholders={sorted(found_placeholders)} in {placeholder_cols} text column(s)")
