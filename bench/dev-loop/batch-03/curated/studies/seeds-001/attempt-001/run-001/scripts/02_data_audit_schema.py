import json
import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["seeds_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]

text_cols = df.select_dtypes(include=["object", "string"]).columns
placeholders = {"?", "-", "NA", "N/A", ""}
found = {}
for c in text_cols:
    vals = df[c].dropna().astype(str).str.strip()
    mask = vals.isin(placeholders) | (vals == "")
    if mask.any():
        found[c] = sorted(set(vals[mask]))
if found:
    print(f"placeholders found in {len(found)} columns: {list(found)}")

schema = pd.DataFrame({
    "name": df.columns,
    "dtype": [str(t) for t in df.dtypes],
    "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
    "n_missing": [int(df[c].isna().sum()) for c in df.columns],
})
if len(df) > 0:
    schema["pct_missing"] = (schema["n_missing"] / len(df) * 100).round(4)
else:
    schema["pct_missing"] = 0.0

mlca_runtime.save_table(schema, "schema.json", caption="Column schema")
mlca_runtime.record_fact_candidate("n_rows", int(len(df)), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(len(schema)), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")
print(f"rows={len(df)} cols={len(schema)} missing={int(schema['n_missing'].sum())}")
