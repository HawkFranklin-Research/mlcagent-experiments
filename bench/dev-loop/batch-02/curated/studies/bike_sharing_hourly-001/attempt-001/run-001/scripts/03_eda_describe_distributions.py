import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['bike_sharing_hourly_uci'])

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing keys {missing} in _protocol. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

exclude_set = set(targets) | set(excluded)
if group_col is not None:
    exclude_set.add(group_col)
feature_cols = [c for c in df.columns if c not in exclude_set]

key_like = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=False) == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

rows = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        rows.append({
            'column': c, 'type': 'numeric',
            'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
            'median': s.median(), 'max': s.max(),
            'n_missing': s.isna().sum(),
            'n_levels': np.nan, 'top_level_share': np.nan
        })
    else:
        vc = s.value_counts(dropna=True)
        top_share = vc.iloc[0] / len(s) if len(vc) > 0 else np.nan
        rows.append({
            'column': c, 'type': 'categorical',
            'mean': np.nan, 'sd': np.nan, 'min': np.nan,
            'median': np.nan, 'max': np.nan, 'n_missing': np.nan,
            'n_levels': s.nunique(dropna=True),
            'top_level_share': top_share
        })

df_summary = pd.DataFrame(rows).round(4)
mlca_runtime.save_table(df_summary, "distributions.csv", caption="Distribution summary of features")

n_numeric = int((df_summary['type'] == 'numeric').sum())
n_categorical = int((df_summary['type'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
placeholders = ['?', '-', 'NA', 'N/A', '']
found_tokens = set()
cols_with_placeholders = 0
for c in text_cols:
    s = df[c].astype(str)
    has_any = False
    for token in placeholders:
        if token == '':
            if s.str.strip().eq('').any():
                found_tokens.add('whitespace')
                has_any = True
        else:
            if (s == token).any():
                found_tokens.add(token)
                has_any = True
    if (s != s.str.strip()).any():
        found_tokens.add('surrounding_whitespace')
        has_any = True
    if has_any:
        cols_with_placeholders += 1

if found_tokens:
    print(f"Placeholders found: {', '.join(sorted(found_tokens))} in {cols_with_placeholders} columns")
else:
    print("No placeholder tokens found.")

print(f"Features: {n_numeric} numeric, {n_categorical} categorical")
print("Saved distributions.csv")
