import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)

ins = mlca_runtime.inputs()
proto = json.loads(ins['_protocol'].read_text())
df = pd.read_parquet(ins['proteome_er'])

targets = proto['targets']
excl = proto.get('excluded_columns', [])
if isinstance(excl, dict):
    excl = excl.get('examples', [])
excl = set(excl)
grp = proto.get('group_column')

feat = [c for c in df.columns if c not in targets and c not in excl and c != grp]

keep = []
for c in feat:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(df):
            continue
    elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == len(df):
            continue
    keep.append(c)
feat = keep

num_cols = [c for c in feat if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in feat if c not in num_cols]
n_num_total = len(num_cols)
n_cat_total = len(cat_cols)


def summarize(sub, c):
    s = sub[c]
    if c in num_cols:
        return dict(type='numeric', mean=float(s.mean()), sd=float(s.std()),
                    mn=float(s.min()), median=float(s.median()), mx=float(s.max()),
                    n_missing=int(s.isna().sum()), n_levels=np.nan, top_share=np.nan)
    vc = s.value_counts(dropna=True, normalize=True)
    return dict(type='categorical', mean=np.nan, sd=np.nan, mn=np.nan, median=np.nan,
                mx=np.nan, n_missing=int(s.isna().sum()),
                n_levels=int(s.nunique(dropna=True)),
                top_share=float(vc.iloc[0]) if len(vc) else np.nan)


groups = [('all', df)]
for t in targets:
    for v in df[t].dropna().unique():
        groups.append((f'{t}={v}', df[df[t] == v]))

n_classes = max(len(groups) - 1, 1)
n_all = max(50 - 5 * n_classes, 10)
all_sel = num_cols[:max(n_all - 8, 1)] + cat_cols[:8]
n_cat_per = 1
n_num_per = max(5 - n_cat_per, 1)
class_sel = num_cols[:n_num_per] + cat_cols[:n_cat_per]

rows = []
for label, sub in groups:
    sel = all_sel if label == 'all' else class_sel
    for c in sel:
        r = summarize(sub, c)
        r['group'] = label
        r['variable'] = c
        rows.append(r)

out = pd.DataFrame(rows).rename(columns={'mn': 'min', 'mx': 'max'})
out = out[['group', 'variable', 'type', 'mean', 'sd', 'min', 'median', 'max',
           'n_missing', 'n_levels', 'top_share']].head(50)
out = out.round(4)

mlca_runtime.save_table(out, 'distributions.csv')
mlca_runtime.record_fact_candidate('n_numeric', n_num_total, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_cat_total, source='distributions.csv')
print('rows', len(out), 'numeric', n_num_total, 'categorical', n_cat_total)
