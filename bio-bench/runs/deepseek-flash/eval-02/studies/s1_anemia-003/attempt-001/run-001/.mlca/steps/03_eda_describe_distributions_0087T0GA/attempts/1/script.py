import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for k in ('targets', 'excluded_columns', 'task'):
    if k not in proto:
        raise KeyError(f"missing protocol key {k}; available: {list(proto.keys())}")
targets = proto['targets']
excluded = proto['excluded_columns'] or []
group_col = proto.get('group_column')

df = pd.read_parquet(inp["anemia"])
print(f"rows={len(df)} cols={df.shape[1]}")

placeholders = {'?', '-', 'NA', 'N/A', ''}
ph_cols = [c for c in df.columns
           if (df[c].dtype == object or str(df[c].dtype) == 'string')
           and df[c].astype(str).str.strip().isin(placeholders).any()]
print(f"placeholder-like tokens in {len(ph_cols)} columns")

feat = [c for c in df.columns
        if c not in targets and c not in excluded and c != group_col]
key_cols = [c for c in feat
            if (df[c].dtype == object or str(df[c].dtype) == 'string')
            and df[c].nunique(dropna=True) == df[c].notna().sum()
            and df[c].nunique(dropna=True) > 1]
feat = [c for c in feat if c not in key_cols]
print(f"features={len(feat)} key_like_dropped={len(key_cols)}")

def is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

def summarize(s, g):
    nm = int(s.isna().sum())
    if is_num(s):
        x = s.dropna().astype(float)
        return dict(type='numeric', group=g, n_missing=nm,
                    mean=x.mean() if len(x) else np.nan,
                    sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                    min=x.min() if len(x) else np.nan,
                    median=x.median() if len(x) else np.nan,
                    max=x.max() if len(x) else np.nan,
                    n_levels=np.nan, top_level=np.nan, top_share=np.nan)
    x = s.dropna().astype(str)
    nl = x.nunique()
    if nl:
        vc = x.value_counts()
        return dict(type='categorical', group=g, n_missing=nm,
                    mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                    n_levels=nl, top_level=vc.index[0], top_share=vc.iloc[0] / len(x))
    return dict(type='categorical', group=g, n_missing=nm,
                mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                n_levels=0, top_level=np.nan, top_share=np.nan)

target = targets[0] if targets else None
groups = [('all', df)]
if target in df.columns:
    for cls in df[target].dropna().unique():
        groups.append((str(cls), df[df[target] == cls]))

rows = []
for c in feat:
    rows.append({'variable': c, **summarize(df[c], 'all')})
    if is_num(df[c]) and len(groups) > 1:
        for g, sub in groups[1:]:
            rows.append({'variable': c, **summarize(sub[c], g)})

cols = ['variable', 'type', 'group', 'mean', 'sd', 'min', 'median', 'max',
        'n_missing', 'n_levels', 'top_level', 'top_share']
out = pd.DataFrame(rows, columns=cols)
if len(out) > 50:
    out = out.head(50).reset_index(drop=True)

for c in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    out[c] = pd.to_numeric(out[c], errors='coerce').apply(
        lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
out['n_missing'] = pd.to_numeric(out['n_missing'], errors='coerce').astype('Int64')
out['n_levels'] = pd.to_numeric(out['n_levels'], errors='coerce').astype('Int64')

mlca_runtime.save_table(out, 'distributions.csv',
                        caption='Per-variable distribution summaries, overall and per target class')

n_num = sum(1 for c in feat if is_num(df[c]))
n_cat = len(feat) - n_num
mlca_runtime.record_fact_candidate('n_numeric', n_num, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_cat, source='distributions.csv')
print(f"n_numeric={n_num} n_categorical={n_cat} rows_saved={len(out)}")
