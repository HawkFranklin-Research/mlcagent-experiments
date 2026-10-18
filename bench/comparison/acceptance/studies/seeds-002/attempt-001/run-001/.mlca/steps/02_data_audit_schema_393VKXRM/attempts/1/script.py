import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["seeds_uci"])
_ = json.loads(inputs["_protocol"].read_text())

placeholder_tokens = {"?", "-", "NA", "N/A"}
placeholder_cols = set()
placeholder_names = set()
rows = []
n_rows = len(df)

for col in df.columns:
    s = df[col]
    dtype = str(s.dtype)
    s_clean = s
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        mask = stripped.isin(placeholder_tokens) | (stripped == "")
        if mask.any():
            placeholder_cols.add(col)
            found = set(stripped[mask].dropna().unique())
            placeholder_names.update([x if x != "" else "whitespace" for x in found])
        s_clean = s.mask(mask)
    n_missing = int(s_clean.isna().sum())
    n_unique = int(s_clean.nunique(dropna=True))
    pct_missing = float(f"{100 * n_missing / n_rows:.4g}") if n_rows else 0.0
    rows.append({
        "name": col,
        "dtype": dtype,
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", len(df.columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

if placeholder_cols:
    print(f"Placeholders found: {', '.join(sorted(placeholder_names))} in {len(placeholder_cols)} columns")
else:
    print("No placeholder tokens found")
print(f"Schema: {len(schema)} columns, {missing_total} missing total")
print(f"Rows: {n_rows}")
