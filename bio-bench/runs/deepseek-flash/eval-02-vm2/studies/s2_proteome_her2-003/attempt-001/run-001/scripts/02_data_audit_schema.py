import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
for key in ("proteome_her2", "_protocol"):
    if key not in paths:
        raise KeyError(f"Missing input {key!r}; available: {sorted(paths)}")

protocol = json.loads(paths["_protocol"].read_text())
df = pd.read_parquet(paths["proteome_her2"])
n_rows = int(df.shape[0])
n_cols = int(df.shape[1])

def round_sig(x, sig=4):
    if pd.isna(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct = (n_missing / n_rows * 100.0) if n_rows else 0.0
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round_sig(pct),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(rows, "schema.json", kind="table")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} cols={n_cols} missing={missing_total}")
