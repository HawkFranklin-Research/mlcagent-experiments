import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["steel_plates_faults_uci"])

placeholder_tokens = {"?", "-", "NA", "N/A"}
found, ph_cols = set(), 0
for col in df.select_dtypes(include=["object", "string"]).columns:
    s = df[col].astype("string").str.strip()
    mask = (s.isin(placeholder_tokens) | (s.str.len() == 0)).fillna(False)
    if mask.any():
        ph_cols += 1
        found.update(s[mask].dropna().unique().tolist())
print(f"placeholders={sorted(found) if found else 'none'} text_columns={ph_cols}")

n = len(df)
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct = float(f"{(n_missing / n * 100) if n else 0.0:.4g}")
    rows.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct,
    })

schema = pd.DataFrame(rows)
mlca_runtime.save_json(schema.to_dict(orient="records"), "schema.json", kind="table")
total_missing = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print(f"rows={df.shape[0]} columns={df.shape[1]} missing_total={total_missing}")
