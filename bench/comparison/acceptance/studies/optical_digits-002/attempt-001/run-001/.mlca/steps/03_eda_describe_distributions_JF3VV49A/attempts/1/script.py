import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for k in ['targets', 'excluded_columns', 'task']:
    if k not in protocol:
        raise ValueError(f"Missing key {k} in protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column', None)

df = pd.read_parquet(inputs['optical_digits_uci'])

exclude_set = set(targets) | set(excluded)
if group_col is not None:
    exclude_set.add(group_col)
for col in df.columns:
    if col in exclude_set:
        continue
    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
        if df[col].nunique(dropna=False) == len(df):
            exclude_set.add(col)

feature_cols = [c for c in df.columns if c not in exclude_set]

text_cols = [c for c in feature_cols if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c])]
placeholder_cols = 0
placeholders_found = set()
for col in text_cols:
    s = df[col].astype(str).str.strip()
    mask = s.isin(['?', '-', 'NA', 'N/A', '']) | (s.str.strip() == '')
    if mask.any():
        placeholder_cols += 1
        placeholders_found.update(s[mask].unique())
print(f"Placeholders found: {placeholders_found if placeholders_found else 'none'} in {placeholder_cols} columns")

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

max_rows = 50
if len(feature_cols) <= max_rows:
    selected_numeric = numeric_cols
    selected_categorical = categorical_cols
else:
    num_var = {c: df[c].var() for c in numeric_cols}
    cat_levels = {c: df[c].nunique(dropna=True) for c in categorical_cols}
    sorted_num = sorted(numeric_cols, key=lambda c: num_var[c], reverse=True)
    sorted_cat = sorted(categorical_cols, key=lambda c: cat_levels[c], reverse=True)
    selected = (sorted_num + sorted_cat)[:max_rows]
    selected_numeric = [c for c in selected if c in numeric_cols]
    selected_categorical = [c for c in selected if c in categorical_cols]

rows = []
for col in selected_numeric:
    s = df[col]
    rows.append({
        'column': col,
        'mean': s.mean(),
        'sd': s.std(),
        'min': s.min(),
        'median': s.median(),
        'max': s.max(),
        'n_missing': int(s.isna().sum()),
        'n_levels': np.nan,
        'top_share': np.nan
    })
for col in selected_categorical:
    s = df[col]
    vc = s.value_counts(dropna=True)
    top_share = vc.iloc[0] / vc.sum() if len(vc) > 0 else np.nan
    rows.append({
        'column': col,
        'mean': np.nan,
        'sd': np.nan,
        'min': np.nan,
        'median': np.nan,
        'max': np.nan,
        'n_missing': int(s.isna().sum()),
        'n_levels': int(s.nunique(dropna=True)),
        'top_share': top_share
    })

summary_df = pd.DataFrame(rows)

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return float(f"{{:.{sig}g}}".format(x))

for col in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    if col in summary_df.columns:
        summary_df[col] = summary_df[col].apply(round_sig)

mlca_runtime.save_table(summary_df, 'distributions.csv', caption='Numeric and categorical feature summaries')

n_numeric = len(selected_numeric)
n_categorical = len(selected_categorical)
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"Rows: {len(df)}, Feature columns: {len(feature_cols)}")
print(f"Summarized: {n_numeric} numeric, {n_categorical} categorical")
