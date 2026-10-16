import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys in _protocol: {missing}. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

df = pd.read_parquet(inputs['anuran_calls_uci'])

if not isinstance(targets, list):
    targets = [targets]
if not isinstance(excluded, list):
    excluded = [excluded]

all_cols = df.columns.tolist()
feature_candidates = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

feature_cols = []
for c in feature_candidates:
    s = df[c]
    n = len(s)
    nunique = s.nunique(dropna=True)
    if nunique == n and (pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s)):
        continue
    feature_cols.append(c)

def sig_round(x, n=4):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, np.integer)):
        return x
    if x == 0:
        return 0.0
    return float(f"{x:.{n}g}")

rows = []
for c in feature_cols:
    s = df[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        kind = 'numeric'
        mean = s.mean()
        sd = s.std()
        minv = s.min()
        median = s.median()
        maxv = s.max()
        n_levels = np.nan
        top_share = np.nan
    else:
        kind = 'categorical'
        mean = sd = minv = median = maxv = np.nan
        n_levels = int(s.nunique(dropna=True))
        top_share = s.value_counts(normalize=True, dropna=True).iloc[0] if n_levels > 0 else np.nan
    rows.append({
        'column': c,
        'kind': kind,
        'n_missing': n_missing,
        'mean': mean,
        'sd': sd,
        'min': minv,
        'median': median,
        'max': maxv,
        'n_levels': n_levels,
        'top_share': top_share,
        'group': np.nan,
        'class': np.nan
    })

for t in targets:
    if t in df.columns:
        vc = df[t].value_counts(dropna=True)
        n_classes = len(vc)
        if len(rows) + n_classes <= 50:
            for cls, count in vc.items():
                share = count / len(df)
                rows.append({
                    'column': t,
                    'kind': 'target_class',
                    'n_missing': np.nan,
                    'mean': np.nan,
                    'sd': np.nan,
                    'min': np.nan,
                    'median': np.nan,
                    'max': np.nan,
                    'n_levels': np.nan,
                    'top_share': share,
                    'group': t,
                    'class': cls
                })

df_summary = pd.DataFrame(rows)
cols = ['column', 'kind', 'n_missing', 'mean', 'sd', 'min', 'median', 'max', 'n_levels', 'top_share', 'group', 'class']
df_summary = df_summary[cols]

if len(df_summary) > 50:
    df_summary = df_summary.head(50)

for c in ['n_missing', 'mean', 'sd', 'min', 'median', 'max', 'n_levels', 'top_share']:
    df_summary[c] = df_summary[c].apply(lambda x: sig_round(x, 4))

mlca_runtime.save_table(df_summary, 'distributions.csv', caption='Distribution summary of feature columns and target classes')

n_numeric = int((df_summary['kind'] == 'numeric').sum())
n_categorical = int((df_summary['kind'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"Feature columns: {len(feature_cols)} (numeric={n_numeric}, categorical={n_categorical})")
print(f"Saved distributions.csv with {len(df_summary)} rows and {len(df_summary.columns)} columns")
