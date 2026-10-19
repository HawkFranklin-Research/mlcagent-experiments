import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing required key '{key}' in _protocol. Available top-level keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

df = pd.read_parquet(inputs['breast_cancer_diagnostic_uci'])

exclude = set(targets) | set(excluded)
if group_col:
    exclude.add(group_col)

feature_cols = []
for c in df.columns:
    if c in exclude:
        continue
    s = df[c]
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(s) and s.nunique() > 1:
        continue
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)

numeric_cols = []
cat_cols = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

rows = []
for c in numeric_cols:
    s = df[c]
    rows.append({
        'feature': c, 'type': 'numeric', 'target_class': 'overall',
        'mean': s.mean(), 'sd': s.std(ddof=1), 'min': s.min(),
        'median': s.median(), 'max': s.max(), 'n_missing': int(s.isna().sum()),
        'n_levels': np.nan, 'top_share': np.nan
    })
for c in cat_cols:
    s = df[c]
    vc = s.value_counts(dropna=True, normalize=True)
    rows.append({
        'feature': c, 'type': 'categorical', 'target_class': 'overall',
        'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan, 'max': np.nan,
        'n_missing': int(s.isna().sum()), 'n_levels': int(s.nunique(dropna=True)),
        'top_share': float(vc.iloc[0]) if len(vc) else np.nan
    })

summary_df = pd.DataFrame(rows)
if len(summary_df) > 50:
    summary_df = summary_df.sort_values(['type', 'feature']).head(50)

for col in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    if col in summary_df.columns:
        summary_df[col] = summary_df[col].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)

mlca_runtime.save_table(summary_df, "distributions.csv", caption="Distribution summaries for numeric and categorical features")

n_numeric = int((summary_df['type'] == 'numeric').sum())
n_categorical = int((summary_df['type'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"Saved distributions.csv with {len(summary_df)} rows")
print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")

text_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
if text_cols:
    placeholders = ['?', '-', 'NA', 'N/A', ' ']
    found = 0
    for c in text_cols:
        s = df[c].astype(str).str.strip()
        if any((s == p).any() for p in placeholders):
            found += 1
    print(f"Placeholder tokens found in {found} text columns")
else:
    print("No text columns for placeholder check")
