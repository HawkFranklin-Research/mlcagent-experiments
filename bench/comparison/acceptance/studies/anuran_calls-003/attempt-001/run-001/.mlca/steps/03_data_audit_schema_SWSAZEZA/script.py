import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["anuran_calls_uci"])
proto = json.loads(paths["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in proto:
        raise KeyError(f"missing protocol key {k!r}; available: {sorted(proto)}")

placeholders = {"?", "-", "NA", "N/A", ""}
hits = {}
for c in df.select_dtypes(include=["object", "string"]).columns:
    s = df[c].astype("string").dropna()
    mask = s.isin(placeholders) | s.str.strip().eq("")
    if mask.any():
        hits[c] = int(mask.sum())
print(
    f"placeholder tokens in {len(hits)} text columns"
    if hits
    else "no placeholder tokens in text columns"
)

rows = []
n = len(df)
for c in df.columns:
    s = df[c]
    n_missing = int(s.isna().sum())
    rows.append(
        {
            "name": c,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": n_missing,
            "pct_missing": round(n_missing / n * 100, 4) if n else np.nan,
        }
    )

schema = pd.DataFrame(rows)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate(
    "missing.total", int(df.isna().sum().sum()), source="schema.json"
)
print(f"shape={df.shape}, missing={int(df.isna().sum().sum())}")
