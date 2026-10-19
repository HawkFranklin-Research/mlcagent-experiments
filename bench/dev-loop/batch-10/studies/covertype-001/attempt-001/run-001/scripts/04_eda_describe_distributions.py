import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['covertype_uci'])

targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns', []) or [])
group_col = protocol.get('group_column')

summary_cols = [c for c in df.columns if c not in excluded and c != group_col]
numeric_cols, categorical_cols = [], []
for c in summary_cols:
    if c in targets or not pd.api.types.is_numeric_dtype(df[c]):
        categorical_cols.append(c)
    else:
        numeric_cols.append(c)

ph = sum(1 for c in categorical_cols
         if df[c].dropna().astype(str).str.strip().isin(['?', '-', 'NA', 'N/A', '']).any())
print(f"placeholder-like tokens in {ph}/{len(categorical_cols)} categorical cols")

def num_from_sub(sub):
    nm = int(sub.isna().sum().sum())
    n = int(sub.notna().all(axis=1).sum())
    arr = sub.to_numpy().ravel()
    arr = arr[~pd.isna(arr)]
    if arr.size == 0:
        return dict(n=0, n_missing=nm, mean=np.nan, sd=np.nan,
                    min=np.nan, median=np.nan, max=np.nan)
    s = pd.Series(arr)
    return dict(n=n, n_missing=nm, mean=float(s.mean()), sd=float(s.std(ddof=0)),
                min=float(s.min()), median=float(s.median()), max=float(s.max()))

def cat_single(col):
    s = df[col]
    nn = int(s.notna().sum())
    if nn == 0:
        return dict(n=0, n_missing=int(s.isna().sum()), mean=np.nan, sd=np.nan,
                    min=np.nan, median=np.nan, max=np.nan, n_levels=0, top_share=np.nan)
    vc = s.value_counts()
    return dict(n=nn, n_missing=int(s.isna().sum()), mean=np.nan, sd=np.nan,
                min=np.nan, median=np.nan, max=np.nan,
                n_levels=int(vc.shape[0]), top_share=float(vc.iloc[0] / nn))

def sig4(x):
    if x is None:
        return x
    try:
        if pd.isna(x):
            return x
    except Exception:
        pass
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

binary, continuous = [], []
for c in numeric_cols:
    v = df[c].dropna()
    if v.isin([0, 1]).all() and v.nunique() <= 2:
        binary.append(c)
    else:
        continuous.append(c)

wild = [c for c in binary if 'wilderness' in c.lower()]
soil = [c for c in binary if 'soil' in c.lower()]
other_bin = [c for c in binary if c not in wild and c not in soil]

rows = []
for c in continuous:
    r = num_from_sub(df[[c]])
    r.update({'group': 'all', 'kind': 'numeric', 'variable': c,
              'n_levels': np.nan, 'top_share': np.nan})
    rows.append(r)
for lbl, cs in [('Wilderness_Area group', wild), ('Soil_Type group', soil), ('Other binaries', other_bin)]:
    if cs:
        r = num_from_sub(df[cs])
        r.update({'group': 'all', 'kind': 'numeric',
                  'variable': f'{lbl} ({len(cs)} cols)', 'n_levels': np.nan, 'top_share': np.nan})
        rows.append(r)
for c in categorical_cols:
    r = cat_single(c)
    r.update({'group': 'all', 'kind': 'categorical', 'variable': c})
    rows.append(r)

for t in targets:
    total = int(df[t].notna().sum())
    for val, cnt in df[t].value_counts().sort_index().items():
        rows.append({'group': str(val), 'kind': 'class_count', 'variable': t,
                     'n': int(cnt), 'n_missing': int(df[t].isna().sum()),
                     'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan, 'max': np.nan,
                     'n_levels': np.nan,
                     'top_share': float(cnt / total) if total else np.nan})
    if continuous:
        for val in sorted(df[t].dropna().unique()):
            r = num_from_sub(df.loc[df[t] == val, continuous])
            r.update({'group': str(val), 'kind': 'numeric_class',
                      'variable': f'(continuous, k={len(continuous)})',
                      'n_levels': np.nan, 'top_share': np.nan})
            rows.append(r)

cols = ['group', 'kind', 'variable', 'n', 'n_missing',
        'mean', 'sd', 'min', 'median', 'max', 'n_levels', 'top_share']
dist = pd.DataFrame(rows)
for c in cols:
    if c not in dist.columns:
        dist[c] = np.nan
dist = dist[cols]
for c in ['mean', 'sd', 'min', 'median', 'max', 'n_levels', 'top_share']:
    dist[c] = dist[c].map(sig4)

mlca_runtime.save_table(dist, 'distributions.csv',
                        caption='EDA distributions: numeric (mean/sd/min/median/max/n_missing) and categorical (n_levels/top_share) summaries overall and per target class')

n_num = int((dist['kind'] == 'numeric').sum())
n_cat = int((dist['kind'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_num, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_cat, source='distributions.csv')
print(f"rows={len(dist)} n_numeric={n_num} n_categorical={n_cat}")
