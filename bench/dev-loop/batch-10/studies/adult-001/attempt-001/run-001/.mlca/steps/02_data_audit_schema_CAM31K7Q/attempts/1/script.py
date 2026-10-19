import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["adult_census_uci"])
json.loads(inputs["_protocol"].read_text())

placeholders = {"?", "-", "NA", "N/A"}
text_cols = [
    c for c in df.columns
    if pd.api.types.is_string_dtype(df[c])
    or pd.api.types.is_object_dtype(df[c])
    or isinstance(df[c].dtype, pd.CategoricalDtype)
]

clean = df.copy()
found_tokens = set()
placeholder_cols = 0

for c in text_cols:
    s = df[c].astype("object")
    vals = s.dropna().astype(str).str.strip()
    mask = vals.eq("") | vals.isin(placeholders)
    if mask.any():
        placeholder_cols += 1
        for v in vals[mask].unique():
            found_tokens.add("whitespace" if v == "" else v)
    stripped = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    cond = stripped.isin(placeholders) | stripped.fillna("").eq("")
    clean[c] = stripped.mask(cond, pd.NA)

rows = []
n = len(clean)
for c in clean.columns:
    nmiss = int(clean[c].isna().sum())
    rows.append({
        "name": c,
        "dtype": str(df[c].dtype),
        "n_unique": int(clean[c].nunique(dropna=True)),
        "n_missing": nmiss,
        "pct_missing": round(100.0 * nmiss / n, 4) if n else float("nan"),
    })

schema = pd.DataFrame(rows)
schema["n_unique"] = schema["n_unique"].astype(int)
schema["n_missing"] = schema["n_missing"].astype(int)
schema["pct_missing"] = schema["pct_missing"].map(
    lambda x: float(f"{x:.4g}") if pd.notna(x) else x
)

mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(len(df)), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

if found_tokens:
    print(f"placeholders found: {sorted(found_tokens)} in {placeholder_cols} columns")
else:
    print("placeholders found: none")
print(f"rows={len(df)} cols={df.shape[1]} missing={total_missing}")
