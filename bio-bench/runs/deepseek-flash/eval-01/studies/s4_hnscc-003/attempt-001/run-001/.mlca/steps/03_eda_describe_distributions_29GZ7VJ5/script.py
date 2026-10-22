import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['discovery'])

targets = list(protocol.get('targets') or [])
excluded = set(protocol.get('excluded_columns') or [])
group_col = protocol.get('group_column')
target = targets[0] if targets else None

feat_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
feat_cols = [c for c in feat_cols if not (
    (df[c].dtype == object or str(df[c].dtype).startswith('string'))
    and df[c].nunique(dropna=True) == len(df) and len(df) > 0)]

numeric_cols, categorical_cols = [], []
n = len(df)
for c in feat_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
    else:
        nu = s.nunique(dropna=True)
        if 1 < nu <= max(20, int(0.05 * n)):
            categorical_cols.append(c)

found = set()
placeholder_cols = 0
for c in categorical_cols:
    s = df[c].astype(str).str.strip()
    hit = False
    for ph in ['?', '-', 'NA', 'N/A', '']:
        if (s == ph).any():
            found.add(ph if ph != '' else 'empty')
            hit = True
    if hit:
        placeholder_cols += 1
print(f"placeholders {sorted(found)} in {placeholder_cols} categorical columns")

def r4(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        return float(f"{v:.4g}")
    except Exception:
        return v

def num_stats(s):
    x = pd.to_numeric(s, errors='coerce')
    k = int(x.notna().sum())
    return dict(n=k, n_missing=int(x.isna().sum()),
                mean=r4(x.mean()) if k else np.nan,
                sd=r4(x.std()) if k > 1 else np.nan,
                min=r4(x.min()) if k else np.nan,
                median=r4(x.median()) if k else np.nan,
                max=r4(x.max()) if k else np.nan)

def cat_stats(s):
    x = s.astype(str).str.strip()
    x = x.mask(x.isin(['?', '-', 'NA', 'N/A', '']), np.nan)
    k = int(x.notna().sum())
    nu = int(x.nunique(dropna=True))
    vc = x.value_counts(dropna=True)
    share = r4(vc.iloc[0] / k) if k and len(vc) else np.nan
    return dict(n=k, n_missing=int(x.isna().sum()), n_levels=nu, top_level_share=share)

classes = [v for v in df[target].dropna().unique().tolist()] if (target and target in df.columns) else []
subsets = [(cls, df[df[target] == cls]) for cls in classes] if classes else [('all', df)]

rows = []
for cls, sub in subsets:
    for c in numeric_cols:
        st = num_stats(sub[c])
        rows.append(dict(variable=c, variable_type='numeric', group=cls, **st,
                         n_levels=np.nan, top_level_share=np.nan))
    for c in categorical_cols:
        st = cat_stats(sub[c])
        rows.append(dict(variable=c, variable_type='categorical', group=cls,
                         n=st['n'], n_missing=st['n_missing'],
                         mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                         n_levels=st['n_levels'], top_level_share=st['top_level_share']))

cols = ['variable','variable_type','group','n','n_missing','mean','sd','min','median','max','n_levels','top_level_share']
dist = pd.DataFrame(rows, columns=cols).head(50)

mlca_runtime.save_table(dist, 'distributions.csv', caption='Distribution summary per target class')

n_numeric = int(dist.loc[dist['variable_type'] == 'numeric', 'variable'].nunique())
n_categorical = int(dist.loc[dist['variable_type'] == 'categorical', 'variable'].nunique())
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')
print(f"saved distributions.csv rows={len(dist)} n_numeric={n_numeric} n_categorical={n_categorical}")
