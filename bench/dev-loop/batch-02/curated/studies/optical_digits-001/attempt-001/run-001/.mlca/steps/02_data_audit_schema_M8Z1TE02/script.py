import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"Missing '{k}'. Available top-level keys: {list(protocol.keys())}")

df = pd.read_parquet(inputs["optical_digits_uci"])

placeholders = {"?", "-", "NA", "N/A", ""}
ph_cols = []
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
        s = df[c].astype("string")
        mask = s.str.strip().isin(placeholders) | s.str.match(r"^\s*$", na=False)
        if mask.any():
            ph_cols.append(c)
print(f"placeholder columns: {len(ph_cols)}")

rows = []
for c in df.columns:
    s = df[c]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / len(df), 4) if len(df) else 0.0,
    })
schema = pd.DataFrame(rows)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print(f"rows={df.shape[0]} cols={df.shape[1]} missing={total_missing}")
