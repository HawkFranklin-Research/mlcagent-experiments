import mlca_runtime, json, pandas as pd, numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['rna_er'])

targets = protocol['targets']
excluded = protocol['excluded_columns']
try:
    group = protocol['group_column']
except KeyError:
    raise KeyError(f"Missing 'group_column' in _protocol. Available keys: {list(protocol.keys())}")

feat_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]

placeholders = ['?', '-', 'NA', 'N/A', '']
placeholder_cols = 0
for c in feat_cols:
    if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c]):
        s = df[c].astype(str).str.strip()
        if s.isin(placeholders).any() or (s == '').any():
            placeholder_cols += 1
            df[c] = s.replace(placeholders, np.nan)
            df[c] = df[c].replace(r'^\s*$', np.nan, regex=True)
print(f"Placeholders found in {placeholder_cols} columns")

numeric_cols = []
categorical_cols = []
drop_cols = []

for c in feat_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        df[c+'_year'] = s.dt.year
        df[c+'_month'] = s.dt.month
        df[c+'_day'] = s.dt.day
        numeric_cols.extend([c+'_year', c+'_month', c+'_day'])
        drop_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if pd.api.types.is_integer_dtype(s) and s.nunique() == len(df):
            drop_cols.append(c)
        else:
            numeric_cols.append(c)
    elif pd.api.types.is_string_dtype(s) or s.dtype == 'object':
        if s.nunique() == len(df):
            drop_cols.append(c)
        elif s.nunique() > 0.05 * len(df):
            drop_cols.append(c)
        else:
            categorical_cols.append(c)
    else:
        drop_cols.append(c)

rows = []
def safe_stat(x, func):
    try:
        return func(x.dropna())
    except:
        return np.nan

for c in numeric_cols:
    s = df[c]
    rows.append({'feature': c, 'type': 'numeric', 'n_missing': int(s.isna().sum()),
                 'mean': safe_stat(s, np.mean), 'sd': safe_stat(s, np.std),
                 'min': safe_stat(s, np.min), 'median': safe_stat(s, np.median),
                 'max': safe_stat(s, np.max)})

for c in categorical_cols:
    s = df[c]
    vc = s.value_counts()
    top_share = vc.iloc[0]/len(s) if len(vc)>0 else np.nan
    rows.append({'feature': c, 'type': 'categorical', 'n_missing': int(s.isna().sum()),
                 'n_levels': int(s.nunique()), 'top_share': top_share})

n_classes = df[targets[0]].nunique() if len(targets)==1 else 0
if len(numeric_cols) <= 10 and n_classes >= 2 and len(rows) + len(numeric_cols)*n_classes <= 50:
    for c in numeric_cols:
        for cls in sorted(df[targets[0]].dropna().unique()):
            sub = df[df[targets[0]]==cls][c]
            rows.append({'feature': c, 'type': 'numeric', 'group': str(cls),
                         'n_missing': int(sub.isna().sum()),
                         'mean': safe_stat(sub, np.mean), 'sd': safe_stat(sub, np.std),
                         'min': safe_stat(sub, np.min), 'median': safe_stat(sub, np.median),
                         'max': safe_stat(sub, np.max)})

dist = pd.DataFrame(rows)
for col in ['mean','sd','min','median','max','top_share']:
    if col in dist.columns:
        dist[col] = dist[col].apply(lambda x: float(f'{x:.4g}') if pd.notna(x) else x)
if len(dist) > 50:
    dist = dist.head(50)
mlca_runtime.save_table(dist, 'distributions.csv', caption='Distribution summary of numeric and categorical features')

n_num = dist[dist['type']=='numeric']['feature'].nunique()
n_cat = dist[dist['type']=='categorical']['feature'].nunique()
mlca_runtime.record_fact_candidate('n_numeric', int(n_num), source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', int(n_cat), source='distributions.csv')

print(f"Saved distributions.csv with {len(dist)} rows; n_numeric={n_num}, n_categorical={n_cat}")
