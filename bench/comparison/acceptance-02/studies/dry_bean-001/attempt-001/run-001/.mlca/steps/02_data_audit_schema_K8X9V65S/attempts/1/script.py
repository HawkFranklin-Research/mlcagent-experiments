import mlca_runtime
import json
import math
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["dry_bean_uci"])
_ = json.loads(inputs["_protocol"].read_text())

def sig4(x):
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return x
    return round(x, 3 - int(math.floor(math.log10(abs(x)))))

n_rows = int(len(df))
n_columns = int(df.shape[1])
records = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": sig4(100.0 * n_missing / n_rows if n_rows else 0.0),
    })

schema = pd.DataFrame(records, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(records, "schema.json", kind="table")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")

placeholder_tokens = {"?", "-", "NA", "N/A"}
found_tokens = set()
ph_cols = 0
pad_cols = 0
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col].dtype) or df[col].dtype == object:
        s = df[col].dropna().astype(str)
        stripped = s.str.strip()
        col_has = False
        for tok in placeholder_tokens:
            if stripped.eq(tok).any():
                found_tokens.add(tok)
                col_has = True
        if stripped.eq("").any():
            found_tokens.add("whitespace-only")
            col_has = True
        if col_has:
            ph_cols += 1
        if (s != stripped).any():
            pad_cols += 1

print(f"rows={n_rows} cols={n_columns} missing_total={total_missing}")
print(f"schema_rows={len(schema)}")
print(f"placeholders={sorted(found_tokens) if found_tokens else 'none'} in {ph_cols} column(s); surrounding_whitespace={pad_cols} column(s)")
