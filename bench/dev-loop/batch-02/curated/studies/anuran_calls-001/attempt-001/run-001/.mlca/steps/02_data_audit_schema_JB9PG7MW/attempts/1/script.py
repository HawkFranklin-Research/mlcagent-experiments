import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["anuran_calls_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys: {missing}. Available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]

placeholder_tokens = {"?", "-", "NA", "N/A", ""}
placeholder_cols = []
for col in df.select_dtypes(include=["object", "string"]).columns:
    s = df[col].astype("string").str.strip()
    mask = s.isin(placeholder_tokens) | s.str.fullmatch(r"\s*", na=False)
    if mask.any():
        placeholder_cols.append(col)
print(f"Placeholders found in {len(placeholder_cols)} text columns: {placeholder_cols[:5]}")

n_rows = len(df)
schema = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = (n_missing / n_rows * 100) if n_rows else 0.0
    schema.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": float(f"{pct_missing:.4g}")
    })
schema_df = pd.DataFrame(schema)

mlca_runtime.save_json(schema_df.to_dict(orient="records"), "schema.json", kind="table")

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", len(df.columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema_df["n_missing"].sum()), source="schema.json")

print(f"rows={n_rows} cols={len(df.columns)} missing={int(schema_df['n_missing'].sum())}")
