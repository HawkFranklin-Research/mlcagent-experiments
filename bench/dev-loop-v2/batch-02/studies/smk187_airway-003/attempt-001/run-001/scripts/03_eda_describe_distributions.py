import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['smk187_airway_openml'])

try:
    targets = protocol['targets']
    excluded = protocol['excluded_columns']
except KeyError as e:
    raise KeyError(f"Missing key {e} in protocol. Available keys: {list(protocol.keys())}")
group_col = protocol.get('group_column')

feature_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if df[c].dtype == 'object' or df[c].dtype.name == 'string':
        if df[c].nunique(dropna=True) == len(df):
            continue
    if pd.api.types.is_integer_dtype(df[c]):
        if df[c].nunique(dropna=True) == len(df) and any(k in c.lower() for k in ['id','index','key']):
            continue
    feature_cols.append(c)

num_cols, cat_cols = [], []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
        num_cols.append(c)
    elif pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = df[c].astype('int64')
        num_cols.append(c)
    else:
        cat_cols.append(c)

placeholder_tokens = ['?', '-', 'NA', 'N/A']
ph_cols = []
for c in cat_cols:
    s = df[c].astype(str).str.strip()
    if any((s == t).any() for t in placeholder_tokens) or (s == '').any():
        ph_cols.append(c)
print(f"Placeholders in {len(ph_cols)} columns: {ph_cols[:3]}" if ph_cols else "No placeholder tokens found.")

groups = {'all': df}
if targets:
    tcol = targets[0]
    for val in df[tcol].dropna().unique():
        groups[f'target={val}'] = df[df[tcol] == val]

records = []
for gname, gdf in groups.items():
    for c in num_cols:
        s = gdf[c]
        records.append({'group': gname, 'feature': c, 'type': 'numeric',
                        'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
                        'median': s.median(), 'max': s.max(), 'n_missing': s.isna().sum()})
    for c in cat_cols:
        s = gdf[c]
        nlev = s.nunique(dropna=True)
        top = s.value_counts(dropna=True, normalize=True).iloc[0] if nlev > 0 else np.nan
        records.append({'group': gname, 'feature': c, 'type': 'categorical',
                        'n_levels': nlev, 'top_share': top})

rec_df = pd.DataFrame(records)
agg_rows = []
for gname in groups:
    for ftype in ['numeric', 'categorical']:
        sub = rec_df[(rec_df['group'] == gname) & (rec_df['type'] == ftype)]
        n_cols = len(sub)
        agg_rows.append({'group': gname, 'feature_type': ftype, 'statistic': 'n_columns',
                         'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan, 'max': np.nan, 'count': n_cols})
        if n_cols == 0:
            continue
        stats = ['mean','sd','min','median','max','n_missing'] if ftype == 'numeric' else ['n_levels','top_share']
        for stat in stats:
            vals = sub[stat].dropna()
            if len(vals) == 0:
                continue
            agg_rows.append({'group': gname, 'feature_type': ftype, 'statistic': stat,
                             'mean': vals.mean(), 'sd': vals.std(), 'min': vals.min(),
                             'median': vals.median(), 'max': vals.max(), 'count': len(vals)})

dist_df = pd.DataFrame(agg_rows)
def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)
for col in ['mean','sd','min','median','max']:
    dist_df[col] = dist_df[col].apply(round_sig)

mlca_runtime.save_table(dist_df, 'distributions.csv', caption='Column distribution summary by target group')

n_numeric = int(dist_df.loc[(dist_df['group']=='all') & (dist_df['feature_type']=='numeric') & (dist_df['statistic']=='n_columns'), 'count'].iloc[0])
n_categorical = int(dist_df.loc[(dist_df['group']=='all') & (dist_df['feature_type']=='categorical') & (dist_df['statistic']=='n_columns'), 'count'].iloc[0])
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"Rows: {len(df)}, features: {len(feature_cols)}")
print(f"Numeric: {n_numeric}, Categorical: {n_categorical}")
print(f"Saved distributions.csv with {len(dist_df)} rows")
