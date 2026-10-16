import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

placeholder_tokens = {"?", "-", "NA", "N/A"}
found_tokens = set()
cols_with_issues = 0
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        stripped = s.str.strip()
        mask_placeholder = stripped.isin(placeholder_tokens) | (stripped == "")
        if mask_placeholder.any():
            found_tokens.update([t for t in stripped[mask_placeholder].unique() if t != ""])
        if (s != stripped).any():
            found_tokens.add("whitespace")
        if mask_placeholder.any() or (s != stripped).any():
            cols_with_issues += 1

rows = []
for col in df.columns:
    n_missing = int(df[col].isna().sum())
    n_unique = int(df[col].nunique(dropna=True))
    pct_missing = float(f"{n_missing / len(df) * 100:.4g}") if len(df) else 0.0
    rows.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })
schema_df = pd.DataFrame(rows)
mlca_runtime.save_table(schema_df, "schema.json")

n_rows = len(df)
n_columns = df.shape[1]
missing_total = int(schema_df["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"Rows: {n_rows}, Columns: {n_columns}")
print(f"Missing total: {missing_total}")
if found_tokens:
    print(f"Placeholders: {', '.join(sorted(found_tokens))} in {cols_with_issues} columns")
else:
    print("Placeholders: none")
