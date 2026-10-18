import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["online_retail_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

n_rows, n_columns = df.shape

def sig4(x):
    if pd.isna(x) or x == 0:
        return float(x)
    return float(f"{x:.4g}")

records = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": sig4(n_missing / n_rows * 100.0) if n_rows else 0.0,
    })

mlca_runtime.save_json(records, "schema.json", kind="table")

missing_total = int(sum(r["n_missing"] for r in records))
mlca_runtime.record_fact_candidate("n_rows", int(n_rows), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(n_columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

placeholder_tokens = {"?", "-", "NA", "N/A", ""}
found_cols = 0
found_tokens = set()
for col in df.columns:
    s = df[col]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.dropna().astype(str).str.strip()
        mask = stripped.isin(placeholder_tokens)
        if mask.any():
            found_cols += 1
            found_tokens.update(stripped[mask].unique())

print(f"rows={n_rows} cols={n_columns}")
print(f"missing_total={missing_total}")
print(f"placeholder_columns={found_cols} tokens={sorted(found_tokens)}")
