import mlca_runtime, json, pandas as pd, numpy as np

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

def g(k):
    if k not in proto:
        raise KeyError(f"missing key '{k}'; available: {list(proto.keys())}")
    return proto[k]

targets = g('targets')
excluded = g('excluded_columns')
task = g('task')
group_col = proto.get('group_column')

df = pd.read_parquet(inp['heart_disease_cleveland_uci'])

skip = set(targets) | set(excluded) | ({group_col} if group_col else set())
feats = [c for c in df.columns if c not in skip]

def is_keylike(s):
    if s.isna().any() or s.nunique() != len(s):
        return False
    return not (pd.api.types.is_bool_dtype(s) or pd.api.types.is_float_dtype(s))

feats = [c for c in feats if not is_keylike(df[c])]

num_feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
cat_feats = [c for c in feats if c not in num_feats]

tgt = targets[0]
classes = [c for c in pd.Series(df[tgt]).dropna().unique()]
groups = [('all', df)] + [(str(c), df[df[tgt] == c]) for c in classes]

rows = []
for var in num_feats:
    for name, sub in groups:
        x = pd.to_numeric(sub[var], errors='coerce')
        d = dict(variable=var, type='numeric', group=name, n=len(sub), n_missing=int(x.isna().sum()))
        if x.notna().any():
            d.update(mean=x.mean(), sd=x.std(), min=x.min(), median=x.median(), max=x.max())
        rows.append(d)

for var in cat_feats:
    for name, sub in groups:
        s = sub[var].dropna()
        d = dict(variable=var, type='categorical', group=name, n=len(sub), n_missing=int(len(sub) - len(s)))
        d['n_levels'] = int(s.nunique())
        if len(s):
            d['top_share'] = float(s.value_counts(normalize=True).iloc[0])
        rows.append(d)

cols = ['variable', 'type', 'group', 'n', 'n_missing', 'mean', 'sd', 'min', 'median', 'max', 'n_levels', 'top_share']
out = pd.DataFrame(rows).reindex(columns=cols)

if len(out) > 50:
    keep = out[out['group'] == 'all']
    room = 50 - len(keep)
    out = pd.concat([keep.head(50)] + ([out[out['group'] != 'all'].head(room)] if room > 0 else []))

for c in ['n', 'n_missing', 'n_levels']:
    out[c] = out[c].astype('Int64')
for c in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    out[c] = out[c].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

mlca_runtime.save_table(out, "distributions.csv", "Numeric and categorical summaries overall and by target class")
mlca_runtime.record_fact_candidate("n_numeric", len(num_feats), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_feats), source="distributions.csv")

print(f"rows={len(df)} features={len(feats)}")
print(f"numeric={len(num_feats)} categorical={len(cat_feats)} summary_rows={len(out)}")
