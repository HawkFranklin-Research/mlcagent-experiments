import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def get_protocol_key(key):
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")
    return protocol[key]

targets = get_protocol_key("targets")
excluded = get_protocol_key("excluded_columns")
group = get_protocol_key("group_column")
task = get_protocol_key("task")

df = pd.read_parquet(inputs["energy_efficiency_uci"])

schema_records = []
for col in df.columns:
    dtype = str(df[col].dtype)
    n_unique = int(df[col].nunique(dropna=True))
    n_missing = int(df[col].isna().sum())
    pct_missing = round(n_missing / len(df) * 100, 4) if len(df) > 0 else 0.0
    schema_records.append({
        "name": col,
        "dtype": dtype,
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing
    })

mlca_runtime.save_json(schema_records, "schema.json", kind="table")

n_rows = len(df)
n_columns = len(df.columns)
missing_total = sum(r["n_missing"] for r in schema_records)

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

placeholders = {"?", "-", "NA", "N/A"}
placeholder_cols = set()
found_placeholders = set()
for col in df.columns:
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        for val in df[col].dropna():
            if isinstance(val, str):
                stripped = val.strip()
                if stripped in placeholders:
                    found_placeholders.add(stripped)
                    placeholder_cols.add(col)
                if val != stripped:
                    placeholder_cols.add(col)

print(f"Rows: {n_rows}, Columns: {n_columns}")
print(f"Total missing: {missing_total}")
if found_placeholders:
    print(f"Placeholders found: {sorted(found_placeholders)} in {len(placeholder_cols)} columns")
else:
    print("No placeholder tokens found in text columns")
