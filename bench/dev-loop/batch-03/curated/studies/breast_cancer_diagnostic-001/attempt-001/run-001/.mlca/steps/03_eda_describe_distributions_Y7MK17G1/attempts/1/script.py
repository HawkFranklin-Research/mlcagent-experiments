import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs['breast_cancer_diagnostic_uci'])
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys in _protocol: {missing}. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
task = protocol['task']

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]

key_like = []
for c in feature_cols:
    if df[c].dtype == 'object' or df[c].dtype == 'string':
        if df[c].nunique() == len(df):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

rows = []
for c in feature_cols:
    col = df[c]
    n_missing = col.isna().sum()
    if pd.api.types.is_numeric_dtype(col):
        typ = 'numeric'
        mean = col.mean()
        sd = col.std()
        minv = col.min()
        median = col.median()
        maxv = col.max()
        n_levels = np.nan
        top_share = np.nan
    else:
        typ = 'categorical'
        mean = sd = minv = median = maxv = np.nan
        n_levels = col.nunique()
        top_share = col.value_counts(normalize=True).iloc[0] if n_levels > 0 else np.nan
    rows.append({
        'column': c, 'group': 'overall', 'type': typ, 'n_missing': n_missing,
        'mean': mean, 'sd': sd, 'min': minv, 'median': median, 'max': maxv,
        'n_levels': n_levels, 'top_share': top_share
    })

max_per_class = (50 - len(feature_cols)) // 2
if max_per_class > 0:
    subset = feature_cols[:max_per_class]
    target_col = targets[0]
    for cls in df[target_col].unique():
        for c in subset:
            col = df[df[target_col] == cls][c]
            n_missing = col.isna().sum()
            if pd.api.types.is_numeric_dtype(col):
                typ = 'numeric'
                mean = col.mean()
                sd = col.std()
                minv = col.min()
                median = col.median()
                maxv = col.max()
                n_levels = np.nan
                top_share = np.nan
            else:
                typ = 'categorical'
                mean = sd = minv = median = maxv = np.nan
                n_levels = col.nunique()
                top_share = col.value_counts(normalize=True).iloc[0] if n_levels > 0 else np.nan
            rows.append({
                'column': c, 'group': str(cls), 'type': typ, 'n_missing': n_missing,
                'mean': mean, 'sd': sd, 'min': minv, 'median': median, 'max': maxv,
                'n_levels': n_levels, 'top_share': top_share
            })

dist_df = pd.DataFrame(rows)

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

for col in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    dist_df[col] = dist_df[col].apply(lambda x: round_sig(x) if pd.notna(x) else x)

mlca_runtime.save_table(dist_df, 'distributions.csv', caption='Distribution summaries')

n_numeric = dist_df[(dist_df['group'] == 'overall') & (dist_df['type'] == 'numeric')].shape[0]
n_categorical = dist_df[(dist_df['group'] == 'overall') & (dist_df['type'] == 'categorical')].shape[0]
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"Features: {len(feature_cols)}")
print(f"n_numeric: {n_numeric}, n_categorical: {n_categorical}")
print(f"distributions.csv rows: {len(dist_df)}")
