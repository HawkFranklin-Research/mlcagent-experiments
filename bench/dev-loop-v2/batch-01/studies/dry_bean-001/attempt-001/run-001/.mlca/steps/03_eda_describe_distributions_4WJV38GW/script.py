import json
import pandas as pd
import numpy as np
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
targets = proto['targets']
excluded = proto['excluded_columns']
group_col = proto.get('group_column')
df = pd.read_parquet(inputs['dry_bean_uci'])

skip = set(targets) | set(excluded)
if group_col:
    skip.add(group_col)

feat_cols = []
for c in df.columns:
    if c in skip:
        continue
    s = df[c]
    if s.nunique(dropna=False) == len(s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
    feat_cols.append(c)

num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in feat_cols if c not in num_cols]

ph_tokens = {'?', '-', 'NA', 'N/A'}
ph_cols = 0
for c in df.columns:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        vals = df[c].dropna().astype(str).str.strip()
        if vals.isin(ph_tokens).any():
            ph_cols += 1

rows = []
for c in num_cols:
    s = pd.to_numeric(df[c], errors='coerce')
    rows.append(dict(column=c, kind='numeric', target_class='',
                     n=int(s.notna().sum()), mean=s.mean(), sd=s.std(),
                     min=s.min(), median=s.median(), max=s.max(),
                     n_missing=int(s.isna().sum()), n_levels=np.nan, top_share=np.nan))
for c in cat_cols:
    s = df[c]
    vc = s.value_counts(dropna=True)
    ts = vc.iloc[0] / len(s) if len(vc) else np.nan
    rows.append(dict(column=c, kind='categorical', target_class='',
                     n=int(s.notna().sum()), mean=np.nan, sd=np.nan,
                     min=np.nan, median=np.nan, max=np.nan,
                     n_missing=int(s.isna().sum()),
                     n_levels=int(s.nunique(dropna=True)), top_share=ts))

for t in targets:
    vc = df[t].value_counts(dropna=True)
    n_total = len(df)
    n_miss = int(df[t].isna().sum())
    for lvl, cnt in vc.items():
        rows.append(dict(column=t, kind='target_level', target_class=str(lvl),
                         n=int(cnt), mean=np.nan, sd=np.nan, min=np.nan,
                         median=np.nan, max=np.nan, n_missing=n_miss,
                         n_levels=int(vc.shape[0]), top_share=cnt / n_total))

out = pd.DataFrame(rows)
for c in ['n', 'mean', 'sd', 'min', 'median', 'max', 'n_missing', 'n_levels', 'top_share']:
    out[c] = pd.to_numeric(out[c], errors='coerce').round(4)
mlca_runtime.save_table(out, 'distributions.csv',
                        caption='Numeric and categorical feature summaries by target class')

n_num = int((out['kind'] == 'numeric').sum())
n_cat = int((out['kind'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_num, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_cat, source='distributions.csv')

print(f'rows={len(df)} numeric={n_num} categorical={n_cat} placeholder_cols={ph_cols}')
