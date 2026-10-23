import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["gli85_glioma_openml"])
protocol = json.loads(paths["_protocol"].read_text())

try:
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
    task = protocol["task"]
    metric = task["primary_metric"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available top-level keys: {list(protocol.keys())}")

for col in targets:
    if col not in df.columns:
        raise KeyError(f"Declared target {col!r} not in table columns")

def is_key_like(s: pd.Series) -> bool:
    if s.dtype == object or str(s.dtype) == "string":
        return s.nunique(dropna=True) == s.notna().sum() and s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s) and not pd.api.types.is_bool_dtype(s)
    return False

features = [c for c in df.columns
            if c not in set(targets) and c not in set(excluded) and not is_key_like(df[c])]

n = len(df)
rows = []
for col in df.columns:
    s = df[col]
    nmiss = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": nmiss,
        "pct_missing": round(100.0 * nmiss / n, 4) if n else 0.0,
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n} columns={df.shape[1]} features={len(features)} targets={targets}")
print(f"missing.total={missing_total} metric={metric}")
print(f"columns with missing={int((schema['n_missing'] > 0).sum())}")
