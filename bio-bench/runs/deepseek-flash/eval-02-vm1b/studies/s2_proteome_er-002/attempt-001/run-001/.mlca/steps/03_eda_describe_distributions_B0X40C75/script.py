import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['proteome_er'])

targets = list(proto.get('targets', []))
exc = proto.get('excluded_columns', [])
if isinstance(exc, dict):
    exc = list(exc.get('examples', []))
group_col = proto.get('group_column')

drop = set(targets) | set(exc)
if group_col:
    drop.add(group_col)

feat_cols = []
for c in df.columns:
    if c in drop:
        continue
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_bool_dtype(s):
        feat_cols.append(c); continue
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and bool(s.notna().all()) and s.nunique() == len(s):
            continue
        feat_cols.append(c); continue
    n_notna = int(s.notna().sum())
    if n_notna and s.nunique() == n_notna:
        continue
    feat_cols.append(c)

num_cols, cat_cols = [], []
for c in feat_cols:
    s = df[c]
    if pd.api.types.is_bool_dtype(s):
        cat_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)
    else:
        cat_cols.append(c)


def num_stats(s):
    if s.notna().any():
        return dict(mean=float(s.mean()), sd=float(s.std()), min=float(s.min()),
                    median=float(s.median()), max=float(s.max()),
                    n_missing=int(s.isna().sum()))
    return dict(mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                n_missing=int(s.isna().sum()))


def cat_stats(s):
    s2 = s.dropna()
    n = len(s2)
    if n == 0:
        return dict(n_levels=0, top_level_share=np.nan, n_missing=int(s.isna().sum()))
    vc = s2.value_counts()
    return dict(n_levels=int(s2.nunique()), top_level_share=float(vc.iloc[0] / n),
                n_missing=int(s.isna().sum()))


def mk(group, col, kind, d):
    return dict(group=group, column=col, kind=kind,
                mean=d.get('mean', np.nan), sd=d.get('sd', np.nan),
                min=d.get('min', np.nan), median=d.get('median', np.nan),
                max=d.get('max', np.nan), n_missing=d.get('n_missing', np.nan),
                n_levels=d.get('n_levels', np.nan),
                top_level_share=d.get('top_level_share', np.nan))

rows = []
for c in num_cols:
    rows.append(mk('all', c, 'numeric', num_stats(df[c])))
for c in cat_cols:
    rows.append(mk('all', c, 'categorical', cat_stats(df[c])))

tgt = targets[0] if targets else None
if tgt and tgt in df.columns:
    classes = df[tgt].dropna().unique().tolist()
    if rows and len(rows) * (1 + len(classes)) <= 50:
        for cls in classes:
            sub = df[df[tgt] == cls]
            for c in num_cols:
                rows.append(mk(f'{tgt}={cls}', c, 'numeric', num_stats(sub[c])))
            for c in cat_cols:
                rows.append(mk(f'{tgt}={cls}', c, 'categorical', cat_stats(sub[c])))

out = pd.DataFrame(rows).head(50)
out = out.round(4)

mlca_runtime.save_table(out, 'distributions.csv',
                        caption='Column distributions: numeric and categorical')
mlca_runtime.record_fact_candidate('n_numeric', int(len(num_cols)),
                                   source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', int(len(cat_cols)),
                                   source='distributions.csv')

print(f"features: numeric={len(num_cols)} categorical={len(cat_cols)}")
print(f"summary_rows={len(out)} targets={targets} group_col={group_col}")
