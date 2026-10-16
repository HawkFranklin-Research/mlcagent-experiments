import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["adult_census_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

# Read protocol keys (not used for schema, but required by rails)
_ = protocol.get("targets", [])
_ = protocol.get("excluded_columns", [])
_ = protocol.get("group_column", None)
_ = protocol.get("task", {})

rows = []
for col in df.columns:
    n_unique = df[col].nunique(dropna=True)
    n_missing = int(df[col].isna().sum())
    pct_missing = round(n_missing / len(df) * 100, 4) if len(df) > 0 else 0.0
    rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

n_rows = len(df)
n_columns = df.shape[1]
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

# Check placeholders in text columns
placeholder_tokens = {'?', '-', 'NA', 'N/A', ''}
found = set()
cols_with_placeholders = 0
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
        s = df[col].dropna().astype(str).str.strip()
        mask = s.isin(placeholder_tokens)
        if mask.any():
            cols_with_placeholders += 1
            found.update(s[mask].unique())

found_display = {('whitespace-only' if t == '' else t) for t in found}
if found_display:
    print(f"Placeholders found: {', '.join(sorted(found_display))} in {cols_with_placeholders} columns.")
else:
    print("No placeholder tokens found.")

print(f"Rows: {n_rows}, Columns: {n_columns}, Missing total: {missing_total}")
