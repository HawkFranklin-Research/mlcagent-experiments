import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)

paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["abalone_uci"])
protocol = json.loads(paths["_protocol"].read_text())

def req(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r} in _protocol; available keys: {sorted(d.keys())}")
    return d[key]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group_col = req(protocol, "group_column")
task = req(protocol, "task")

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

def is_keylike(s):
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s) and s.nunique(dropna=True) > 0
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=True) == len(s) and s.nunique(dropna=True) > 0
    return False

feature_cols = [c for c in df.columns if c not in drop and not is_keylike(df[c])]

placeholders = {'?', '-', 'NA', 'N/A', ''}
placeholder_cols = []
for c in feature_cols:
    if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
        s = df[c].astype('string').str.strip()
        mask = s.isin(placeholders)
        if mask.any():
            placeholder_cols.append(c)
            df[c] = s.mask(mask, pd.NA)
        else:
            df[c] = s

rows = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        rows.append({
            'variable': c, 'type': 'numeric', 'group': 'all',
            'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
            'median': s.median(), 'max': s.max(),
            'n_missing': int(s.isna().sum()),
            'n_levels': np.nan, 'top_share': np.nan
        })
    else:
        vc = s.value_counts(dropna=True)
        n_levels = int(vc.size)
        top_share = float(vc.iloc[0] / s.notna().sum()) if n_levels > 0 else np.nan
        rows.append({
            'variable': c, 'type': 'categorical', 'group': 'all',
            'mean': np.nan, 'sd': np.nan, 'min': np.nan,
            'median': np.nan, 'max': np.nan,
            'n_missing': int(s.isna().sum()),
            'n_levels': n_levels, 'top_share': top_share
        })

summary = pd.DataFrame(rows)
for c in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    summary[c] = summary[c].apply(lambda x: float(f'{x:.4g}') if pd.notna(x) else np.nan)

mlca_runtime.save_table(summary, "distributions.csv", caption="Column distributions")

n_numeric = int((summary['type'] == 'numeric').sum())
n_categorical = int((summary['type'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"rows={len(summary)} cols={len(summary.columns)}")
print(f"n_numeric={n_numeric} n_categorical={n_categorical}")
print(f"placeholders in {len(placeholder_cols)} columns: {placeholder_cols}")
