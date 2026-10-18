import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
if not isinstance(protocol, dict):
    raise TypeError("_protocol must contain a JSON object")

df = pd.read_parquet(inputs["adult_census_uci"])
n_rows = int(df.shape[0])
n_columns = int(df.shape[1])

def sig4(x):
    if pd.isna(x):
        return float("nan")
    return float(f"{float(x):.4g}")

def found_placeholders(s):
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return set()
    vals = s.dropna()
    if len(vals) == 0:
        return set()
    stripped = vals.astype(str).str.strip()
    found = set()
    for tok in ("?", "-", "NA", "N/A"):
        if (stripped == tok).any():
            found.add(tok)
    if (stripped == "").any():
        found.add("<whitespace>")
    return found

schema = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = (100.0 * n_missing / n_rows) if n_rows else float("nan")
    schema.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": sig4(pct_missing),
    })

mlca_runtime.save_json(schema, "schema.json", kind="table")
missing_total = int(sum(row["n_missing"] for row in schema))

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

placeholder_cols = 0
placeholder_tokens = set()
for col in df.columns:
    found = found_placeholders(df[col])
    if found:
        placeholder_cols += 1
        placeholder_tokens |= found

print(f"rows={n_rows} columns={n_columns}")
print(f"missing.total={missing_total}")
if placeholder_cols:
    print(f"placeholder tokens in {placeholder_cols} text columns: {', '.join(sorted(placeholder_tokens))}")
else:
    print("no placeholder tokens in text columns")
