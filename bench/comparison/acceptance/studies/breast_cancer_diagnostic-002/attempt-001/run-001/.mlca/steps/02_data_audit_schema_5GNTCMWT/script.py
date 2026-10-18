import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing '{key}' in _protocol; available: {sorted(proto)}")

df = pd.read_parquet(paths["breast_cancer_diagnostic_uci"])

placeholders = {"?", "-", "na", "n/a", "nan", "none", ""}
ph_cols = []
for col in df.columns:
    if df[col].dtype == object or str(df[col].dtype).startswith("string"):
        vals = df[col].dropna().astype(str)
        stripped = vals.str.strip()
        if stripped.str.lower().isin(placeholders).any() or (vals != stripped).any():
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
schema = pd.DataFrame(rows)

n_rows = int(df.shape[0])
n_columns = int(df.shape[1])
missing_total = int(df.isna().sum().sum())

mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} cols={n_columns} missing_total={missing_total}")
