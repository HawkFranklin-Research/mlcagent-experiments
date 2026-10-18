import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing '{key}' in _protocol; available top-level keys: {sorted(proto)}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
task = proto["task"]
group = proto.get("group_column")

df = pd.read_parquet(paths["breast_cancer_diagnostic_uci"])

placeholders = {"?", "-", "na", "n/a", "nan", "none", ""}
ph_cols = []
for col in df.columns:
    if df[col].dtype == object or str(df[col].dtype).startswith("string"):
        vals = df[col].dropna().astype(str)
        if vals.str.strip().str.lower().isin(placeholders).any() or (vals != vals.str.strip()).any():
            ph_cols.append(col)
print(f"text placeholder/whitespace columns: {len(ph_cols)}")

rows = []
for col in df.columns:
    s = df[col]
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": int(s.isna().sum()),
        "pct_missing": round(float(s.isna().mean() * 100.0), 4),
    })

n_rows, n_columns = int(df.shape[0]), int(df.shape[1])
missing_total = int(sum(r["n_missing"] for r in rows))

mlca_runtime.save_json(
    {"n_rows": n_rows, "n_columns": n_columns, "missing_total": missing_total, "columns": rows},
    "schema.json",
    kind="table",
)
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns}")
print(f"missing_total={missing_total}")
print(f"targets={targets} excluded={len(excluded)} group={group}")
