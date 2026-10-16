import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["covertype_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing_keys = [k for k in required if k not in protocol]
if missing_keys:
    raise KeyError(f"Missing protocol keys: {missing_keys}. Available: {list(protocol.keys())}")

n_rows = len(df)
n_columns = len(df.columns)
rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    n_unique = int(df[col].nunique(dropna=True))
    pct_missing = float(f"{n_missing / n_rows * 100:.4g}") if n_rows > 0 else 0.0
    rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })
schema_df = pd.DataFrame(rows)
mlca_runtime.save_table(schema_df, "schema.json", caption="Column schema")

missing_total = int(schema_df["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

placeholders_found = set()
placeholder_cols = set()
for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
        s = df[col].dropna().astype(str)
        stripped = s.str.strip()
        if (s != stripped).any():
            placeholders_found.add("surrounding whitespace")
            placeholder_cols.add(col)
        for ph in ["?", "-", "NA", "N/A"]:
            if (stripped == ph).any():
                placeholders_found.add(ph)
                placeholder_cols.add(col)
        if (stripped == "").any():
            placeholders_found.add("whitespace-only")
            placeholder_cols.add(col)

print(f"Rows: {n_rows}, Columns: {n_columns}")
print(f"Total missing: {missing_total}")
if placeholders_found:
    print(f"Placeholders found: {placeholders_found} in {len(placeholder_cols)} columns")
else:
    print("No placeholders found")
print("Schema saved to schema.json")
