import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['anuran_calls_uci'])

for key in ['targets', 'excluded_columns', 'group_column']:
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in protocol. Available keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

feature_cols = []
for col in df.columns:
    if col in targets or col in excluded or col == group_col:
        continue
    s = df[col]
    if s.dtype == 'object' and s.nunique(dropna=True) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(col)

numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_feature_cols = [c for c in feature_cols if c not in numeric_cols]
cat_cols = cat_feature_cols + [t for t in targets if t in df.columns]

rows = []
for col in numeric_cols:
    s = df[col].dropna()
    rows.append({
        'column': col, 'type': 'numeric',
        'n_missing': int(df[col].isna().sum()),
        'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
        'median': s.median(), 'max': s.max(),
        'n_levels': np.nan, 'top_share': np.nan
    })
for col in cat_cols:
    s = df[col]
    vc = s.value_counts(dropna=True)
    rows.append({
        'column': col, 'type': 'categorical',
        'n_missing': int(s.isna().sum()),
        'mean': np.nan, 'sd': np.nan, 'min': np.nan,
        'median': np.nan, 'max': np.nan,
        'n_levels': int(s.nunique(dropna=True)),
        'top_share': float(vc.iloc[0] / len(s)) if len(vc) > 0 else np.nan
    })

dist = pd.DataFrame(rows)
for col in dist.select_dtypes(include=[np.number]).columns:
    dist[col] = dist[col].apply(lambda x: float(f"{x:.4g}") if pd.notnull(x) else x)

placeholder_tokens = ['?', '-', 'NA', 'N/A']
ph_counts = {}
for col in df.select_dtypes(include=['object']).columns:
    s = df[col].astype(str).str.strip()
    for tok in placeholder_tokens:
        cnt = (s == tok).sum()
        if cnt > 0:
            ph_counts[tok] = ph_counts.get(tok, 0) + 1
    if (s == '').sum() > 0:
        ph_counts['whitespace'] = ph_counts.get('whitespace', 0) + 1
if ph_counts:
    print("Placeholders found: " + ", ".join(f"{k} in {v} columns" for k, v in ph_counts.items()))
else:
    print("No placeholder tokens found in text columns.")

mlca_runtime.save_table(dist, "distributions.csv",
                        caption="Distribution summary of numeric features and categorical targets")
n_numeric = len(numeric_cols)
n_categorical = len(cat_cols)
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"Rows: {len(df)}, feature cols: {len(feature_cols)}")
print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
