import mlca_runtime
import json, numpy as np, pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
for k in ['targets', 'excluded_columns']:
    if k not in proto:
        raise ValueError(f"Protocol missing '{k}'; available: {list(proto.keys())}")
targets = proto['targets']
excluded = proto['excluded_columns']
group = proto.get('group_column', None)
df = pd.read_parquet(inp['gcm_tumour_types_openml'])

features = [c for c in df.columns if c not in targets and c not in excluded and c != group]
keep = []
for c in features:
    s = df[c]
    if s.is_unique and (pd.api.types.is_object_dtype(s) or
                        (pd.api.types.is_integer_dtype(s) and not set(s.dropna().unique()).issubset({0, 1}))):
        continue
    keep.append(c)
features = keep
numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
cat = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])
       and not pd.api.types.is_datetime64_any_dtype(df[c])]
for c in features:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c + '_elapsed'] = (df[c] - df[c].min()).dt.total_seconds()
        numeric.append(c + '_elapsed')

ph = {'?', '-', 'NA', 'N/A', ''}
found = 0
for c in cat:
    s = df[c].astype('string').str.strip()
    m = s.isin(ph) & df[c].notna()
    if m.any():
        found += 1
        df[c] = s.mask(m)
if found:
    print(f"placeholder tokens found in {found} columns")

rows = []
if numeric:
    st = df[numeric].agg(['mean', 'std', 'min', 'median', 'max']).T
    rows.append(dict(scope='numeric', name='all_features', n_rows=len(df), n_columns=len(numeric),
                     mean=st['mean'].median(), sd=st['std'].median(), min=st['min'].min(),
                     median=st['median'].median(), max=st['max'].max(),
                     n_missing=int(df[numeric].isna().sum().sum()), n_levels=np.nan, top_share=np.nan))
else:
    rows.append(dict(scope='numeric', name='all_features', n_rows=len(df), n_columns=0,
                     mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_missing=0, n_levels=np.nan, top_share=np.nan))

for t in targets:
    if t not in df.columns:
        continue
    vc = df[t].value_counts(dropna=False)
    rows.append(dict(scope='categorical', name=t, n_rows=len(df), n_columns=1,
                     mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                     n_missing=int(df[t].isna().sum()), n_levels=int(df[t].nunique(dropna=True)),
                     top_share=float(vc.iloc[0] / len(df)) if len(vc) else np.nan))
    if numeric:
        for cls, sub in df.groupby(t, dropna=False):
            st = sub[numeric].agg(['mean', 'std', 'min', 'median', 'max']).T
            rows.append(dict(scope='target_class', name=str(cls), n_rows=len(sub), n_columns=len(numeric),
                             mean=st['mean'].median(), sd=st['std'].median(), min=st['min'].min(),
                             median=st['median'].median(), max=st['max'].max(),
                             n_missing=int(sub[numeric].isna().sum().sum()), n_levels=np.nan,
                             top_share=len(sub) / len(df)))

dist = pd.DataFrame(rows)
for c in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    dist[c] = dist[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
for c in ['n_rows', 'n_columns', 'n_missing', 'n_levels']:
    dist[c] = dist[c].astype('Int64')
mlca_runtime.save_table(dist, "distributions.csv", caption="Summary of numeric features and target distribution")

n_numeric = len(numeric)
n_cat = len(cat) + sum(1 for t in targets if t in df.columns and not pd.api.types.is_numeric_dtype(df[t]))
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_cat, source='distributions.csv')

print(f"numeric={n_numeric} categorical={n_cat}")
print(f"distributions rows={len(dist)} cols={dist.shape[1]}")
print("saved distributions.csv")
