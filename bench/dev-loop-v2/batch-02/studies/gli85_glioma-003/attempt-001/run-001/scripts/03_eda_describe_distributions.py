import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'task']:
    if k not in proto:
        raise KeyError(f"protocol missing '{k}'; available: {sorted(proto.keys())}")
df = pd.read_parquet(paths['gli85_glioma_openml'])

targets = list(proto['targets'])
excluded = list(proto.get('excluded_columns') or [])
group_col = proto.get('group_column')
drop_set = set(targets) | set(excluded)
if group_col:
    drop_set.add(group_col)

placeholders = {'?', '-', 'NA', 'N/A', ''}


def clean_text(series):
    s = series.astype('string').str.strip()
    return s.mask(s.isin(placeholders))


numeric_cols, cat_cols, dt_cols = [], [], []
for c in df.columns:
    if c in drop_set:
        continue
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and s.notna().all() and s.nunique() == len(s):
            if s.max() - s.min() == len(s) - 1 and s.min() in (0, 1):
                continue
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

tmp = []
for c in cat_cols:
    s = clean_text(df[c]).dropna()
    if len(s) > 5 and s.nunique() == len(s):
        continue
    tmp.append(c)
cat_cols = tmp

ph_cols = 0
for c in cat_cols:
    s = df[c].astype('string').str.strip()
    if s.isin(placeholders).sum() > 0:
        ph_cols += 1
if ph_cols:
    print(f"placeholders {sorted(placeholders - {''})}: {ph_cols} columns affected")

num_df = df[numeric_cols].astype('Float64').astype('float64') if numeric_cols else pd.DataFrame(index=df.index)
for c in dt_cols:
    dt = pd.to_datetime(df[c], errors='coerce')
    num_df[f'{c}__hour'] = dt.dt.hour.astype('float64')
    num_df[f'{c}__dow'] = dt.dt.dayofweek.astype('float64')
    num_df[f'{c}__month'] = dt.dt.month.astype('float64')

cat_df = pd.DataFrame({c: clean_text(df[c]) for c in cat_cols}, index=df.index) if cat_cols else pd.DataFrame(index=df.index)
cat_targets = [t for t in targets if not pd.api.types.is_numeric_dtype(df[t])]
cat_all_df = pd.DataFrame({c: clean_text(df[c]) for c in cat_cols + cat_targets}, index=df.index)


def num_rows(frame, group):
    if frame.shape[1] == 0:
        return []
    stats = pd.DataFrame({
        'mean': frame.mean(), 'sd': frame.std(ddof=1),
        'min': frame.min(), 'median': frame.median(), 'max': frame.max(),
        'n_missing': frame.isna().sum(),
    })
    out = []
    for stat in ['mean', 'sd', 'min', 'median', 'max']:
        v = stats[stat].dropna()
        if len(v):
            out.append({'group': group, 'type': 'numeric', 'statistic': stat, 'n_features': int(len(v)),
                        'min': v.min(), 'q25': v.quantile(.25), 'median': v.median(),
                        'q75': v.quantile(.75), 'max': v.max(), 'mean': v.mean()})
    share = stats['n_missing'] / max(len(frame), 1)
    if len(share):
        out.append({'group': group, 'type': 'numeric', 'statistic': 'n_missing_share', 'n_features': int(len(share)),
                    'min': share.min(), 'q25': share.quantile(.25), 'median': share.median(),
                    'q75': share.quantile(.75), 'max': share.max(), 'mean': share.mean()})
    return out


def cat_rows(frame, group):
    if frame.shape[1] == 0:
        return []
    lv, tp = [], []
    for c in frame.columns:
        s = frame[c]
        lv.append(float(s.nunique(dropna=True)))
        vc = s.value_counts(dropna=True, normalize=True)
        tp.append(float(vc.iloc[0]) if len(vc) else np.nan)
    out = []
    for nm, vals in [('n_levels', lv), ('top_share', tp)]:
        s = pd.Series(vals).dropna()
        if len(s):
            out.append({'group': group, 'type': 'categorical', 'statistic': nm, 'n_features': int(len(s)),
                        'min': s.min(), 'q25': s.quantile(.25), 'median': s.median(),
                        'q75': s.quantile(.75), 'max': s.max(), 'mean': s.mean()})
    return out


rows = num_rows(num_df, 'overall') + cat_rows(cat_all_df, 'overall')
target = targets[0]
for cls, idx in df.groupby(target).groups.items():
    rows += num_rows(num_df.loc[idx], f'{target}={cls}')
    rows += cat_rows(cat_df.loc[idx], f'{target}={cls}')

dist = pd.DataFrame(rows)
cols = ['group', 'type', 'statistic', 'n_features', 'min', 'q25', 'median', 'q75', 'max', 'mean']
dist = dist.reindex(columns=cols).head(50)
for c in ['min', 'q25', 'median', 'q75', 'max', 'mean']:
    dist[c] = dist[c].map(lambda v: v if pd.isna(v) else float(f"{v:.4g}"))
dist['n_features'] = dist['n_features'].astype(int)

mlca_runtime.save_table(dist, "distributions.csv",
                        caption="EDA per-column statistic distributions overall and by target class")

n_numeric = int(num_df.shape[1])
n_categorical = int(len(cat_cols))
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, unit='columns', source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, unit='columns', source='distributions.csv')

print(f"rows={len(df)}; features: numeric={n_numeric}, categorical={n_categorical}")
print(f"distributions.csv: rows={len(dist)}, cols={dist.shape[1]}, groups={dist['group'].nunique()}")
