import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing '{key}' in _protocol; available keys: {list(protocol)}")

df = pd.read_parquet(inputs["optical_digits_uci"])
df = df.reset_index(drop=True)

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(float(n_missing) / max(len(df), 1) * 100.0, 4),
    })
schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])

mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")

n_rows = int(len(df))
n_cols = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

obj_cols = [c for c in df.columns if df[c].dtype == object]
placeholders = {"?", "-", "NA", "N/A", ""}
n_ph_cols = sum(
    1 for c in obj_cols
    if df[c].astype("string").str.strip().isin(placeholders).any()
)
print(f"shape={df.shape}")
print(f"missing.total={missing_total}")
print(f"text_columns={len(obj_cols)} placeholder_columns={n_ph_cols}")
