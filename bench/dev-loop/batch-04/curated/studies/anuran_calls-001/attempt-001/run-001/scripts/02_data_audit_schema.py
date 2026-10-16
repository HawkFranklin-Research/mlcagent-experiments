import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs['anuran_calls_uci'])
protocol = json.loads(inputs['_protocol'].read_text())

if not isinstance(protocol, dict):
    raise ValueError("Protocol JSON is not a dict")

placeholders = {'?', '-', 'NA', 'N/A'}
found_placeholders = set()
placeholder_cols = set()
for col in df.select_dtypes(include=['object', 'string']).columns:
    s = df[col].astype('string')
    stripped = s.str.strip()
    mask_ph = s.notna() & stripped.isin(placeholders)
    mask_ws = s.notna() & (stripped == '')
    mask_surround = s.notna() & (s != stripped)
    if mask_ph.any() or mask_ws.any() or mask_surround.any():
        placeholder_cols.add(col)
    for ph in placeholders:
        if (stripped == ph).any():
            found_placeholders.add(ph)
    if mask_ws.any():
        found_placeholders.add('whitespace-only')
    if mask_surround.any():
        found_placeholders.add('surrounding whitespace')

if placeholder_cols:
    print(f"Placeholders found: {', '.join(sorted(found_placeholders))} in {len(placeholder_cols)} columns")
else:
    print("No placeholder tokens or surrounding whitespace found in text columns.")

n_rows = len(df)
n_cols = len(df.columns)
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    pct_missing = (n_missing / n_rows * 100) if n_rows > 0 else 0.0
    pct_missing = float(f"{pct_missing:.4g}")
    rows.append({
        'name': col,
        'dtype': str(s.dtype),
        'n_unique': n_unique,
        'n_missing': n_missing,
        'pct_missing': pct_missing
    })

schema_df = pd.DataFrame(rows)
mlca_runtime.save_table(schema_df, "schema.json", caption="Column schema")

missing_total = int(schema_df['n_missing'].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"Rows: {n_rows}, Columns: {n_cols}")
print(f"Total missing: {missing_total}")
