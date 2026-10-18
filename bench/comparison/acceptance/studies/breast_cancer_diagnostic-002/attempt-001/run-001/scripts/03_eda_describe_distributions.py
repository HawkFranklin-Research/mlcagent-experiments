import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ['targets', 'excluded_columns', 'task', 'group_column']:
    if key not in protocol:
        raise KeyError(f"Missing '{key}' in protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']

df = pd.read_parquet(inputs['breast_cancer_diagnostic_uci'])

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]

# Drop key-like text columns: all distinct non-missing strings
key_like = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        if s.notna().all() and s.nunique() == len(s):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

# RL-0023: detect placeholders, strip whitespace, convert to missing
placeholder_tokens = {'?', '-', 'NA', 'N/A', ''}
placeholder_cols = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        stripped = s.astype(str).str.strip()
        mask = stripped.isin(placeholder_tokens) & s.notna()
        if mask.any():
            placeholder_cols.append(c)
        df[c] = stripped.replace(list(placeholder_tokens), np.nan)

if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {sorted(placeholder_tokens)}")
else:
    print("No placeholder tokens found in text columns.")

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

rows = []
for c in numeric_cols:
    s = pd.to_numeric(df[c], errors='coerce')
    rows.append({
        'variable': c,
        'type': 'numeric',
        'target_class': 'overall',
        'mean': s.mean(),
        'sd': s.std(),
        'min': s.min(),
        'median': s.median(),
        'max': s.max(),
        'n_missing': int(s.isna().sum()),
        'n_levels': np.nan,
        'top_share': np.nan,
    })

for c in cat_cols:
    s = df[c]
    vc = s.value_counts(dropna=True)
    n_levels = int(vc.shape[0])
    top_share = float(vc.iloc[0] / vc.sum()) if n_levels else np.nan
    rows.append({
        'variable': c,
        'type': 'categorical',
        'target_class': 'overall',
        'mean': np.nan,
        'sd': np.nan,
        'min': np.nan,
        'median': np.nan,
        'max': np.nan,
        'n_missing': int(s.isna().sum()),
        'n_levels': n_levels,
        'top_share': top_share,
    })

summary = pd.DataFrame(rows)
if len(summary) > 50:
    summary = summary.head(50)

for col in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    if col in summary.columns:
        summary[col] = summary[col].apply(lambda x: float(f'{x:.4g}') if pd.notna(x) else np.nan)

mlca_runtime.save_table(summary, 'distributions.csv', caption='Distribution summary by feature')

n_numeric = len(numeric_cols)
n_categorical = len(cat_cols)
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, unit='count', source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, unit='count', source='distributions.csv')

print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
print(f"distributions.csv shape={summary.shape}")
