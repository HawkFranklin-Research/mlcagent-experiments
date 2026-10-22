import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns', []))
group = protocol.get('group_column')

df = pd.read_parquet(inputs['abalone_uci'])

blocked = set(targets) | set(excluded)
feature_cols = [c for c in df.columns if c not in blocked and c != group]


def is_text(s):
    return (pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype))


# drop key-like columns: text with all distinct values, or integer id-like sequences
key_like = []
for c in list(feature_cols):
    s = df[c]
    if is_text(s) and s.notna().sum() and s.nunique(dropna=True) == s.notna().sum():
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.notna().all() and s.nunique() == len(s):
        key_like.append(c)
for c in key_like:
    feature_cols.remove(c)

# placeholder token check on text columns
placeholders = {'?', '-', 'NA', 'N/A', ''}
ph_cols = 0
for c in feature_cols:
    s = df[c]
    if is_text(s):
        stripped = s.astype('string').str.strip()
        hits = int(stripped.isin(placeholders).sum())
        if hits > 0:
            ph_cols += 1
print(f"placeholder tokens found in {ph_cols} text columns")

rows = []
n_numeric = 0
n_categorical = 0
for c in feature_cols:
    s = df[c]
    n_miss = int(s.isna().sum())
    if pd.api.types.is_datetime64_any_dtype(s):
        rows.append({'variable': c, 'kind': 'datetime', 'n': int(s.notna().sum()),
                     'n_missing': n_miss, 'mean': np.nan, 'sd': np.nan, 'min': np.nan,
                     'median': np.nan, 'max': np.nan, 'n_levels': np.nan, 'top_share': np.nan})
    elif pd.api.types.is_numeric_dtype(s):
        rows.append({'variable': c, 'kind': 'numeric', 'n': int(s.notna().sum()),
                     'n_missing': n_miss, 'mean': float(s.mean()), 'sd': float(s.std(ddof=1)),
                     'min': float(s.min()), 'median': float(s.median()), 'max': float(s.max()),
                     'n_levels': np.nan, 'top_share': np.nan})
        n_numeric += 1
    else:
        clean = s.dropna().astype('string').str.strip()
        clean = clean.mask(clean.isin(placeholders))
        vc = clean.dropna().value_counts(normalize=True)
        top = float(vc.iloc[0]) if len(vc) else np.nan
        rows.append({'variable': c, 'kind': 'categorical', 'n': int(clean.notna().sum()),
                     'n_missing': n_miss, 'mean': np.nan, 'sd': np.nan, 'min': np.nan,
                     'median': np.nan, 'max': np.nan,
                     'n_levels': int(clean.nunique(dropna=True)), 'top_share': top})
        n_categorical += 1

summary = pd.DataFrame(rows).round(4)
if len(summary) > 50:
    summary = summary.head(50)

mlca_runtime.save_table(summary, 'distributions.csv',
                        caption='Per-feature distribution summary (numeric and categorical)')
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical} key_like_dropped={len(key_like)}")
print(f"summary rows={len(summary)} cols={summary.shape[1]}")
