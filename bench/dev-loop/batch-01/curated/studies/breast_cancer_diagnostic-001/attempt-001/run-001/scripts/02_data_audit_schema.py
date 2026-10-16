import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs['breast_cancer_diagnostic_uci'])
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(f"Missing keys in _protocol: {missing}. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']
task = protocol['task']

n_rows = len(df)
exclude_set = set(targets) | set(excluded)
if group is not None:
    exclude_set.add(group)

key_like = []
for col in df.columns:
    if col in exclude_set:
        continue
    if df[col].dtype == 'object' or pd.api.types.is_integer_dtype(df[col]):
        if df[col].nunique() == n_rows:
            key_like.append(col)

feature_cols = [c for c in df.columns if c not in exclude_set and c not in key_like]

schema_rows = []
for col in df.columns:
    dtype = str(df[col].dtype)
    n_unique = int(df[col].nunique(dropna=True))
    n_missing = int(df[col].isna().sum())
    pct_missing = round(n_missing / n_rows * 100, 4) if n_rows > 0 else 0.0
    schema_rows.append({
        'name': col,
        'dtype': dtype,
        'n_unique': n_unique,
        'n_missing': n_missing,
        'pct_missing': pct_missing
    })

schema_df = pd.DataFrame(schema_rows)

mlca_runtime.save_json(schema_df.to_dict(orient='records'), "schema.json", kind='table')

mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", len(df.columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema_df['n_missing'].sum()), source="schema.json")

placeholder_tokens = ['?', '-', 'NA', 'N/A']
placeholder_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        series = df[col].astype(str).str.strip()
        found = False
        for token in placeholder_tokens:
            if (series == token).any():
                found = True
                break
        if not found and (series == '').any():
            found = True
        if found:
            placeholder_cols.append(col)

if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {placeholder_cols[:5]}")
else:
    print("No placeholder tokens found in text columns.")

print(f"Rows: {n_rows}, Columns: {len(df.columns)}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Total missing: {int(schema_df['n_missing'].sum())}")
