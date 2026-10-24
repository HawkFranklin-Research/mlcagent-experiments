import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['discovery'])

def get_key(d, key):
    if key not in d:
        raise KeyError(f"Missing key '{key}'. Available top-level keys: {list(d.keys())}")
    return d[key]

targets = get_key(protocol, 'targets')
excluded = get_key(protocol, 'excluded_columns')
task = get_key(protocol, 'task')
group_col = protocol.get('group_column')
if group_col is not None and group_col not in df.columns:
    group_col = None

features = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]

key_like = []
for c in features:
    s = df[c]
    if s.dtype == 'object' and s.nunique() == len(df) and len(df) > 1:
        key_like.append(c)
    elif s.dtype.kind in 'iu' and s.nunique() == len(df) and len(df) > 1:
        if s.max() - s.min() + 1 == len(df):
            key_like.append(c)
features = [c for c in features if c not in key_like]

numeric_features = df[features].select_dtypes(include='number').columns.tolist()
categorical_features = [c for c in features if c not in numeric_features]

n_numeric = len(numeric_features)
n_categorical = len(categorical_features)

placeholders = {'?', '-', 'NA', 'N/A', ''}
placeholder_cols = 0
for c in categorical_features:
    vals = df[c].dropna().astype(str).str.strip()
    if vals.isin(placeholders).any() or (vals == '').any():
        placeholder_cols += 1
print(f"Placeholder tokens found in {placeholder_cols} categorical columns")

summary_numeric = numeric_features[:20]
summary_categorical = categorical_features[:20]
summary_cols = summary_numeric + summary_categorical + targets

rows = []
for col in summary_cols:
    if col not in df.columns:
        continue
    s = df[col]
    is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    kind = 'numeric' if is_num else 'categorical'
    n_missing = int(s.isna().sum())
    if kind == 'numeric':
        rows.append({
            'variable': col, 'kind': kind, 'group': 'all',
            'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
            'median': s.median(), 'max': s.max(),
            'n_missing': n_missing, 'n_levels': np.nan, 'top_share': np.nan
        })
    else:
        vc = s.value_counts(normalize=True, dropna=True)
        top_share = float(vc.iloc[0]) if len(vc) > 0 else np.nan
        rows.append({
            'variable': col, 'kind': kind, 'group': 'all',
            'mean': np.nan, 'sd': np.nan, 'min': np.nan,
            'median': np.nan, 'max': np.nan,
            'n_missing': n_missing, 'n_levels': int(s.nunique(dropna=True)),
            'top_share': top_share
        })

if len(targets) >= 2:
    duration = targets[0]
    event = targets[1]
    if duration in df.columns and event in df.columns and pd.api.types.is_numeric_dtype(df[duration]):
        for g in df[event].dropna().unique():
            sub = df[df[event] == g][duration]
            rows.append({
                'variable': duration, 'kind': 'numeric', 'group': str(g),
                'mean': sub.mean(), 'sd': sub.std(), 'min': sub.min(),
                'median': sub.median(), 'max': sub.max(),
                'n_missing': int(sub.isna().sum()),
                'n_levels': np.nan, 'top_share': np.nan
            })

dist_df = pd.DataFrame(rows)
for c in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    if c in dist_df.columns:
        dist_df[c] = dist_df[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float, np.number)) else x)
if len(dist_df) > 50:
    dist_df = dist_df.head(50)

mlca_runtime.save_table(dist_df, 'distributions.csv', caption='Summary distributions of numeric and categorical columns')
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')
print(f"Saved distributions.csv with {len(dist_df)} rows")
print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
