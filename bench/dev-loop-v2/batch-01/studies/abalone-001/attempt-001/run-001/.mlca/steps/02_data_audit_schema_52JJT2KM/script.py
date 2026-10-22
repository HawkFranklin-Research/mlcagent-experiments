import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
_protocol = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["abalone_uci"])

n_rows, n_cols = df.shape
placeholders = {"?", "-", "NA", "N/A"}
found = set()
cols_with = 0
for col in df.columns:
    s = df[col]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        vals = s.dropna().astype(str).str.strip()
        hit = vals[vals.isin(placeholders) | vals.eq("")]
        if len(hit):
            cols_with += 1
            found.update(vals[vals.isin(placeholders)].unique())
            if vals.eq("").any():
                found.add("whitespace")
if found:
    print("placeholder tokens found: " + ", ".join(sorted(repr(x) for x in found)) + f" in {cols_with} columns")
else:
    print("placeholder tokens found: none (0 columns)")

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct = 0.0 if n_rows == 0 else 100.0 * n_missing / n_rows
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": float(f"{pct:.4g}"),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(rows, "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_cols), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

print(f"rows={n_rows} columns={n_cols} missing_total={total_missing}")
print("saved schema.json")
