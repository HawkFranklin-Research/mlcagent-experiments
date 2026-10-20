import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
top = list(protocol.keys())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Protocol missing required key '{key}'. Available keys: {top}")

df = pd.read_parquet(inp["breast_cancer_diagnostic_uci"])
n = len(df)
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": float(round(100.0 * n_missing / n, 4)) if n else float("nan"),
    })
schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(n), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n} cols={df.shape[1]} missing_total={missing_total}")
